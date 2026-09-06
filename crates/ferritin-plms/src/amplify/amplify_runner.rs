//! Amplify RUnner
//!
//! Class for loading and running the AMPLIFY models

use super::super::types::{ContactMap, PseudoProbability};
use super::amplify::{AMPLIFY, AmplifyOutput};
use super::config::AMPLIFYConfig;
use crate::loader::{LoadOptions, WeightSource};
use crate::plm_runner::{ModelMetadata, PlmRunner, SpecialTokenLayout};
use anyhow::{Error as E, Result, anyhow};
use candle_core::{D, Device, Tensor};
use candle_nn::ops;
use tokenizers::Tokenizer;

pub enum AmplifyModels {
    AMP120M,
    AMP350M,
}
impl AmplifyModels {
    pub fn get_model_files(model: Self) -> WeightSource {
        match model {
            AmplifyModels::AMP120M => {
                WeightSource::safetensors("chandar-lab/AMPLIFY_120M").at_revision("main")
            }
            AmplifyModels::AMP350M => {
                WeightSource::safetensors("chandar-lab/AMPLIFY_350M").at_revision("main")
            }
        }
    }
}

pub struct AmplifyRunner {
    model: AMPLIFY,
    tokenizer: Tokenizer,
}
impl AmplifyRunner {
    pub fn load_model(modeltype: AmplifyModels, device: Device) -> Result<AmplifyRunner> {
        let source = AmplifyModels::get_model_files(modeltype);
        let config_filename = source.fetch("config.json")?;
        let tokenizer_filename = source.fetch("tokenizer.json")?;
        let config_str = std::fs::read_to_string(config_filename)?;
        let config_str = config_str
            .replace("SwiGLU", "swiglu")
            .replace("Swiglu", "swiglu");
        let config: AMPLIFYConfig = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = source.var_builder("model.safetensors", &LoadOptions::new(device))?;
        let model = AMPLIFY::load(vb, &config)?;
        Ok(AmplifyRunner { model, tokenizer })
    }
    pub fn run_forward(&self, prot_sequence: &str) -> Result<AmplifyOutput> {
        let device = self.model.get_device();
        let tokens = self
            .tokenizer
            .encode(prot_sequence.to_string(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let encoded = self.model.forward(&token_ids, None, false, true)?;
        Ok(encoded)
    }
    pub fn get_best_prediction(
        &self,
        prot_sequence: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let model_output: AmplifyOutput = self.run_forward(prot_sequence)?;
        let predictions = model_output.logits.argmax(D::Minus1)?;
        let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
        let decoded = self.tokenizer.decode(indices.as_slice(), true)?;
        let decoded = decoded.replace(" ", "");
        Ok(decoded)
    }
    /// Per-residue pseudo-probabilities over the full AMPLIFY vocabulary.
    ///
    /// `position` is the residue index: BOS and EOS rows are stripped first,
    /// so there is exactly one position per input residue. Previously these
    /// rows were included, which both over-counted positions by two and
    /// shifted every label by one — position 0 was BOS, not the first residue
    /// (ferritin-100.18).
    ///
    /// Unlike [`ESM2Runner::get_pseudo_probabilities`], every amino acid is
    /// returned rather than only those above a probability threshold.
    pub fn get_pseudo_probabilities(&self, prot_sequence: &str) -> Result<Vec<PseudoProbability>> {
        let model_output: AmplifyOutput = self.run_forward(prot_sequence)?;
        let layout = <Self as PlmRunner>::special_tokens(self);
        let residue_rows = model_output.logits.dim(1)?.saturating_sub(layout.total());
        let predictions = model_output
            .logits
            .narrow(1, layout.leading, residue_rows)?;
        let outputs = self.extract_logits(&predictions)?;
        Ok(outputs)
    }
    pub fn get_contact_map(&self, prot_sequence: &str) -> Result<Vec<ContactMap>> {
        let model_output: AmplifyOutput = self.run_forward(prot_sequence)?;
        let contact_map_tensor = model_output.get_contact_map()?.ok_or_else(|| {
            anyhow!("AMPLIFY forward() returned no attentions for the contact map")
        })?;
        let averaged = contact_map_tensor.max_keepdim(D::Minus1)?;
        let (position1, position2, val) = averaged.dims3()?;
        let data = averaged.to_vec3::<f32>()?;

        // Per-position residue labels. The contact map is BOS/EOS-stripped, so
        // position `p` is the p-th residue; decode the actual token ids the
        // model saw (encode adds BOS/EOS, so the residues are ids[1..len-1])
        // rather than decoding the position index as if it were a token id.
        let encoded = self
            .tokenizer
            .encode(prot_sequence.to_string(), true)
            .map_err(E::msg)?;
        let ids = encoded.get_ids();
        let residue_ids = if ids.len() >= 2 {
            &ids[1..ids.len() - 1]
        } else {
            ids
        };
        let label_at = |p: usize| -> char {
            residue_ids
                .get(p)
                .and_then(|&id| self.tokenizer.decode(&[id], true).ok())
                .and_then(|s| s.chars().next())
                .unwrap_or('?')
        };

        let mut contacts = Vec::new();
        #[allow(clippy::needless_range_loop)] // i/j/k index a 3-D contact tensor
        for i in 0..position1 {
            for j in 0..position2 {
                for k in 0..val {
                    contacts.push(ContactMap {
                        position_1: i,
                        amino_acid_1: label_at(i),
                        position_2: j,
                        amino_acid_2: label_at(j),
                        contact_estimate: data[i][j][k],
                        layer: 1,
                    });
                }
            }
        }
        Ok(contacts)
    }
    /// Softmax `tensor` and flatten it into per-(position, amino acid) rows.
    ///
    /// `tensor` must already have its special-token rows stripped, so
    /// `seq_pos` is a residue index.
    fn extract_logits(&self, tensor: &Tensor) -> Result<Vec<PseudoProbability>> {
        let tensor = ops::softmax(tensor, D::Minus1)?;
        let data = tensor.to_vec3::<f32>()?;
        let (_, seq_len, vocab_size) = tensor.dims3()?;
        let mut logit_positions = Vec::with_capacity(seq_len * vocab_size);
        #[allow(clippy::needless_range_loop)] // indexes a 3-D logits tensor
        for seq_pos in 0..seq_len {
            for vocab_idx in 0..vocab_size {
                let score = data[0][seq_pos][vocab_idx];
                let amino_acid_char = self
                    .tokenizer
                    .decode(&[vocab_idx as u32], false)
                    .map_err(|e| anyhow!("Failed to decode: {}", e))?
                    .chars()
                    .next()
                    .ok_or_else(|| anyhow!("Empty decoded string"))?;
                logit_positions.push(PseudoProbability {
                    position: seq_pos,
                    amino_acid: amino_acid_char,
                    pseudo_prob: score,
                });
            }
        }
        Ok(logit_positions)
    }
}

impl PlmRunner for AmplifyRunner {
    /// Run AMPLIFY and return the last-layer hidden states as per-residue embeddings.
    ///
    /// Shape: `(1, L, hidden_size)` where `L` includes BOS and EOS tokens.
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        let device = self.model.get_device();
        let tokens = self
            .tokenizer
            .encode(sequence.to_string(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let output = self.model.forward(&token_ids, None, true, false)?;
        let mut hidden_states = output
            .hidden_states
            .ok_or_else(|| anyhow!("AMPLIFY forward() returned no hidden states"))?;
        hidden_states
            .pop()
            .ok_or_else(|| anyhow!("AMPLIFY returned empty hidden states list"))
    }

    fn model_name(&self) -> &str {
        "amplify"
    }

    /// AMPLIFY's `tokenizer.json` has a `TemplateProcessing` post-processor
    /// that adds `<bos>` (3) and `<eos>` (4), so `encode(.., true)` wraps the
    /// sequence.
    fn special_tokens(&self) -> SpecialTokenLayout {
        SpecialTokenLayout::BOS_EOS
    }

    fn metadata(&self) -> ModelMetadata {
        let config = self.model.config();
        ModelMetadata {
            d_model: config.hidden_size,
            n_layers: config.num_hidden_layers,
            vocab_size: config.vocab_size,
            max_positions: Some(config.max_length),
        }
    }

    fn device(&self) -> &Device {
        self.model.get_device()
    }

    /// Masked-LM logits `(1, L + 2, vocab_size)`.
    fn logits(&self, sequence: &str) -> Result<Tensor> {
        Ok(self.run_forward(sequence)?.logits)
    }
}
