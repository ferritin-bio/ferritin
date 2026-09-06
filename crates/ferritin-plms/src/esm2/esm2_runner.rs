//! ESM2 Runner
//!
//! Class for loading and running the ESM2 models
use super::esm2::{ESM2, ESM2Config, ESM2Output};
use crate::loader::{LoadOptions, WeightSource};
use crate::plm_runner::PlmRunner;
use crate::types::PseudoProbability;
use anyhow::{Error as E, Result, anyhow};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use serde_json;
use tokenizers::Tokenizer;

// ESM2 tokenizer: indices 4-23 are the 20 standard amino acids.
const ESM2_STD_AA: [(usize, char); 20] = [
    (4, 'L'),
    (5, 'A'),
    (6, 'G'),
    (7, 'V'),
    (8, 'S'),
    (9, 'E'),
    (10, 'R'),
    (11, 'T'),
    (12, 'I'),
    (13, 'D'),
    (14, 'P'),
    (15, 'K'),
    (16, 'Q'),
    (17, 'N'),
    (18, 'F'),
    (19, 'Y'),
    (20, 'M'),
    (21, 'H'),
    (22, 'W'),
    (23, 'C'),
];

pub enum ESM2Models {
    T6_8M,
    T12_35M,
    T30_150M,
    T33_650M,
    T36_3B,
    T48_15B,
}
impl ESM2Models {
    pub fn get_model_files(model: Self) -> (WeightSource, ESM2Config) {
        let (repo, config) = match model {
            Self::T6_8M => ("facebook/esm2_t6_8M_UR50D", ESM2Config::t6_8m()),
            Self::T12_35M => ("facebook/esm2_t12_35M_UR50D", ESM2Config::t12_35m()),
            Self::T30_150M => ("facebook/esm2_t30_150M_UR50D", ESM2Config::t30_150m()),
            Self::T33_650M => ("facebook/esm2_t33_650M_UR50D", ESM2Config::t33_650m()),
            Self::T36_3B => ("facebook/esm2_t36_3B_UR50D", ESM2Config::t36_3b()),
            Self::T48_15B => ("facebook/esm2_t48_15B_UR50D", ESM2Config::t48_15b()),
        };
        (WeightSource::safetensors(repo).at_revision("main"), config)
    }
}

pub struct ESM2Runner {
    model: ESM2,
    tokenizer: Tokenizer,
}
impl ESM2Runner {
    /// Load model from HuggingFace hub, downloading config.json, tokenizer files, and weights.
    pub fn load_model(modeltype: ESM2Models, device: Device) -> Result<ESM2Runner> {
        let (source, fallback_config) = ESM2Models::get_model_files(modeltype);
        // Try to load config from HF hub; fall back to hardcoded config if unavailable.
        let config = match source.fetch_optional("config.json") {
            Some(config_path) => {
                let config_str = std::fs::read_to_string(config_path)?;
                serde_json::from_str::<ESM2Config>(&config_str).unwrap_or(fallback_config)
            }
            None => fallback_config,
        };
        let vb = source.var_builder("model.safetensors", &LoadOptions::new(device))?;
        let model = ESM2::load(vb, config)?;
        let tokenizer = ESM2::load_tokenizer()?;
        Ok(ESM2Runner { model, tokenizer })
    }
    /// Encode a sequence to token ids **with** ESM-2's BOS (`<cls>`) and EOS
    /// (`<eos>`) special tokens.
    ///
    /// The bundled `tokenizer.json` has no post-processor, so `encode(.., true)`
    /// would still NOT add them. ESM-2 is trained with BOS/EOS wrapping the
    /// sequence, and every consumer here (`predict_contacts`,
    /// `get_pseudo_probabilities`, `embed`, `decode_logits`) assumes an `L + 2`
    /// row output and strips those two rows. Running the model without them
    /// feeds a different context and its logits diverge from the HF reference,
    /// so they must be added explicitly.
    fn encode_with_special(&self, sequence: &str) -> Result<Vec<u32>> {
        let bos = self
            .tokenizer
            .token_to_id("<cls>")
            .ok_or_else(|| anyhow!("ESM2 tokenizer missing <cls> token"))?;
        let eos = self
            .tokenizer
            .token_to_id("<eos>")
            .ok_or_else(|| anyhow!("ESM2 tokenizer missing <eos> token"))?;
        let inner = self
            .tokenizer
            .encode(sequence.to_string(), false)
            .map_err(E::msg)?;
        let mut ids = Vec::with_capacity(inner.get_ids().len() + 2);
        ids.push(bos);
        ids.extend_from_slice(inner.get_ids());
        ids.push(eos);
        Ok(ids)
    }
    pub fn run_forward(&self, prot_sequence: &str) -> Result<ESM2Output> {
        let device = self.model.get_device();
        let tokens = self.encode_with_special(prot_sequence)?;
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let encoded = self.model.forward(&token_ids, None)?;
        Ok(encoded)
    }
    /// Predict residue-residue contact probabilities for a single protein sequence.
    ///
    /// Returns a `(seq_len, seq_len)` contact probability matrix (BOS/EOS stripped,
    /// so dimensions equal the number of amino acids in `prot_sequence`).
    pub fn predict_contacts(&self, prot_sequence: &str) -> Result<Tensor> {
        let device = self.model.get_device();
        let tokens = self.encode_with_special(prot_sequence)?;
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        // squeeze batch dim: (1, L, L) → (L, L)
        self.model
            .predict_contacts(&token_ids, None)
            .map_err(E::msg)?
            .squeeze(0)
            .map_err(E::msg)
    }

    pub fn decode_logits(&self, output: ESM2Output) -> Result<String> {
        // Get the predicted token IDs by taking argmax along the vocabulary dimension
        let predicted_token_ids = output.logits.argmax(2)?;
        let predicted_token_ids = if predicted_token_ids.dims().len() > 1 {
            predicted_token_ids.squeeze(0)?
        } else {
            predicted_token_ids
        };
        let token_ids: Vec<u32> = predicted_token_ids.to_vec1::<u32>()?;
        let decoded_sequence = self
            .tokenizer
            .decode(&token_ids, true) // set skip_special_tokens to true
            .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?
            .replace(" ", "");
        Ok(decoded_sequence)
    }

    /// Run ESM2 and return per-residue pseudo-probabilities for the 20 standard amino acids.
    ///
    /// BOS and EOS tokens are stripped before softmax is applied. Only amino acid / position
    /// pairs with probability > 0.01 are included in the result.
    pub fn get_pseudo_probabilities(&self, prot_sequence: &str) -> Result<Vec<PseudoProbability>> {
        let output = self.run_forward(prot_sequence)?;
        // logits: (1, L+2, vocab_size) — strip BOS at 0 and EOS at -1
        let seq_len = output.logits.dim(1)? - 2;
        let logits = output.logits.narrow(1, 1, seq_len)?;
        let probs = softmax(&logits, 2)?;
        let probs = probs.squeeze(0)?; // (L, vocab)
        let probs_data: Vec<Vec<f32>> = probs.to_vec2()?;

        let mut result = Vec::new();
        for (pos, pos_probs) in probs_data.iter().enumerate() {
            for (vocab_idx, aa_char) in ESM2_STD_AA.iter() {
                let prob = pos_probs[*vocab_idx];
                if prob > 0.01 {
                    result.push(PseudoProbability {
                        position: pos,
                        pseudo_prob: prob,
                        amino_acid: *aa_char,
                    });
                }
            }
        }
        Ok(result)
    }
}

impl PlmRunner for ESM2Runner {
    /// Run the ESM2 transformer and return per-residue embeddings (pre-LM-head).
    ///
    /// Shape: `(1, L, hidden_size)` where `L` includes BOS and EOS tokens.
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        let device = self.model.get_device();
        let tokens = self.encode_with_special(sequence)?;
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        Ok(self.model.embed(&token_ids, None)?)
    }

    fn model_name(&self) -> &str {
        "esm2"
    }
}
