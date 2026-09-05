use crate::esmc::layers::regression_head::RegressionHead;
use crate::esmc::layers::transformer_stack::TransformerStack;
use crate::esmc::tokenization::TokenizerCollection;
use crate::esmc::tokenization::sequence_tokenizer::EsmSequenceTokenizer;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Output of the ESMC forward pass.
#[derive(Debug)]
pub struct ESMCOutput {
    /// Per-residue sequence logits of shape `(B, L, vocab_size)`.
    pub sequence_logits: Tensor,
    /// Final hidden states of shape `(B, L, d_model)`.
    pub embeddings: Option<Tensor>,
    /// Hidden states from every transformer layer `(n_layers, B, L, d_model)`,
    /// present only when `output_hidden_states = true`.
    pub hidden_states: Option<Tensor>,
}

/// Configuration for the `logits()` wrapper.
#[derive(Debug, Default)]
pub struct LogitsConfig {
    /// Include sequence logits in the output.
    pub sequence: bool,
    /// Include the final embedding tensor.
    pub return_embeddings: bool,
    /// Include per-layer hidden states.
    pub return_hidden_states: bool,
}

/// Output of the `logits()` wrapper.
#[derive(Debug)]
pub struct LogitsOutput {
    /// Sequence logits `(B, L, vocab_size)`, present when `LogitsConfig::sequence = true`.
    pub sequence_logits: Option<Tensor>,
    /// Final embeddings `(B, L, d_model)`, present when `LogitsConfig::return_embeddings = true`.
    pub embeddings: Option<Tensor>,
    /// Per-layer hidden states, present when `LogitsConfig::return_hidden_states = true`.
    pub hidden_states: Option<Tensor>,
}

// ---------------------------------------------------------------------------
// Tokenizer enum
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum ESMTokenizer {
    Esm3OpenSmall,
}

impl ESMTokenizer {
    pub fn get_model_tokenizers(&self) -> TokenizerCollection {
        match self {
            ESMTokenizer::Esm3OpenSmall => TokenizerCollection {
                sequence: EsmSequenceTokenizer::default(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// FFN type
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum FfnType {
    SWIGLU,
    GLU,
}

// ---------------------------------------------------------------------------
// Model config
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ESMCConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub v_head_transformer: Option<usize>,
    pub ffn_type: FfnType,
    pub tokenizer: ESMTokenizer,
    pub use_plain_attn: bool,
    pub n_layers_geom: usize,
    pub scale_residue: bool,
    pub residue_scaling_factor: f64,
    pub mask_and_zero_frameless: bool,
    pub bias: bool,
    pub qk_layernorm: bool,
    pub expansion_ratio: f64,
    // regression head dims
    pub regression_head_output_dim: usize,
    pub regression_head_hidden_dim: usize,
    pub embedding_dim: usize,
}

impl ESMCConfig {
    pub fn esmc_300m() -> Self {
        Self {
            d_model: 960,
            n_heads: 15,
            n_layers: 30,
            v_head_transformer: None,
            ffn_type: FfnType::SWIGLU,
            tokenizer: ESMTokenizer::Esm3OpenSmall,
            use_plain_attn: true,
            n_layers_geom: 0,
            scale_residue: true,
            residue_scaling_factor: (30f64 / 36.).sqrt(),
            mask_and_zero_frameless: false,
            bias: false,
            qk_layernorm: true,
            expansion_ratio: 8.0 / 3.0,
            regression_head_output_dim: 64,
            regression_head_hidden_dim: 960,
            embedding_dim: 64,
        }
    }

    pub fn esmc_600m() -> Self {
        Self {
            d_model: 1152,
            n_heads: 18,
            n_layers: 36,
            v_head_transformer: None,
            ffn_type: FfnType::SWIGLU,
            tokenizer: ESMTokenizer::Esm3OpenSmall,
            use_plain_attn: true,
            n_layers_geom: 0,
            scale_residue: true,
            // Mirrors the sqrt(n_layers / 36) scaling of the other sizes; with
            // n_layers = 36 this is exactly 1.0 (written directly so clippy's
            // eq_op lint doesn't fire on 36 / 36).
            residue_scaling_factor: 1.0,
            mask_and_zero_frameless: false,
            bias: false,
            qk_layernorm: true,
            expansion_ratio: 8.0 / 3.0,
            regression_head_output_dim: 64,
            regression_head_hidden_dim: 1152,
            embedding_dim: 64,
        }
    }

    pub fn esmc_6b() -> Self {
        Self {
            d_model: 2560,
            n_heads: 40,
            n_layers: 80,
            v_head_transformer: None,
            ffn_type: FfnType::SWIGLU,
            tokenizer: ESMTokenizer::Esm3OpenSmall,
            use_plain_attn: true,
            n_layers_geom: 0,
            scale_residue: true,
            residue_scaling_factor: (80f64 / 36.).sqrt(),
            mask_and_zero_frameless: false,
            bias: false,
            qk_layernorm: true,
            expansion_ratio: 8.0 / 3.0,
            regression_head_output_dim: 64,
            regression_head_hidden_dim: 2560,
            embedding_dim: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct ESMC {
    embed: candle_nn::Embedding,
    transformer: TransformerStack,
    sequence_head: RegressionHead,
    tokenizer: EsmSequenceTokenizer,
    device: Device,
}

impl ESMC {
    pub fn load(vb: VarBuilder, config: ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            tokenizer,
            embedding_dim,
            ..
        } = config.clone();

        let device = vb.device().clone();
        let tokenizer_collection = tokenizer.get_model_tokenizers();

        Ok(Self {
            embed: nn::embedding(embedding_dim, d_model, vb.pp("embed"))?,
            transformer: TransformerStack::load(vb.pp("transformer"), &config)?,
            sequence_head: RegressionHead::load(vb.pp("sequence_head"), &config)?,
            tokenizer: tokenizer_collection.sequence,
            device,
        })
    }

    // ---------------------------------------------------------------------------
    // Core forward pass
    // ---------------------------------------------------------------------------

    /// Forward pass of the ESMC model.
    ///
    /// # Arguments
    /// * `sequence_tokens` – token IDs of shape `(B, L)` (u32 integers).
    /// * `sequence_id` – optional boolean mask of shape `(B, L)` where `true`
    ///   means the position is a real (non-pad) token.  When `None`, all
    ///   positions are treated as real tokens.
    /// * `output_hidden_states` – when `true`, every transformer layer's
    ///   output is stacked into `ESMCOutput::hidden_states`.
    pub fn forward(
        &self,
        sequence_tokens: &Tensor,
        sequence_id: Option<&Tensor>,
        output_hidden_states: bool,
    ) -> Result<ESMCOutput> {
        use crate::esmc::tokenization::sequence_tokenizer::EsmTokenizerBase;

        // Build a boolean mask from padding tokens when not supplied.
        let owned_mask;
        let sequence_id = match sequence_id {
            Some(s) => s,
            None => {
                let pad_id = self.tokenizer.pad_token_id();
                owned_mask = sequence_tokens.ne(pad_id)?;
                &owned_mask
            }
        };

        // Embed tokens: (B, L) → (B, L, d_model)
        let x = self.embed.forward(sequence_tokens)?;

        // Transformer stack
        let (x, hidden_states) =
            self.transformer
                .forward(&x, Some(sequence_id), output_hidden_states)?;

        // Stack hidden states along a new leading dimension: (n_layers, B, L, d_model)
        let hidden_states = hidden_states.map(|hs| Tensor::stack(&hs, 0)).transpose()?;

        // Sequence head: (B, L, d_model) → (B, L, vocab_size)
        let sequence_logits = self.sequence_head.forward(&x)?;

        Ok(ESMCOutput {
            sequence_logits,
            embeddings: Some(x),
            hidden_states,
        })
    }

    // ---------------------------------------------------------------------------
    // Convenience wrappers
    // ---------------------------------------------------------------------------

    /// Tokenize a raw amino-acid sequence string into a 1-D token-ID tensor
    /// (shape `(L+2,)` including BOS/EOS), suitable for batching or direct
    /// use as the `sequence_tokens` argument to `forward()`.
    pub fn encode(&self, sequence: &str) -> Result<Tensor> {
        let token_ids = self.tokenizer.tokenize_sequence(sequence, true);
        let len = token_ids.len();
        Tensor::from_vec(token_ids, len, &self.device)
    }

    /// Decode a 1-D or 2-D token-ID tensor back to an amino-acid string.
    ///
    /// Handles a batched `(1, L)` tensor by squeezing the batch dimension.
    /// Special tokens (BOS, PAD, EOS, MASK) are stripped.
    pub fn decode(&self, tokens: &Tensor) -> Result<String> {
        let tokens = match tokens.dims().len() {
            2 => tokens.squeeze(0)?,
            _ => tokens.clone(),
        };
        let ids = tokens.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        Ok(self.tokenizer.decode_sequence(&ids))
    }

    /// Run a no-grad forward pass and return logits / embeddings.
    ///
    /// `tokens` should be a `(B, L)` or `(L,)` tensor of token IDs.
    /// A missing batch dimension is added automatically.
    pub fn logits(&self, tokens: &Tensor, config: LogitsConfig) -> Result<LogitsOutput> {
        // Ensure batch dimension.
        let tokens = match tokens.dims().len() {
            1 => tokens.unsqueeze(0)?,
            _ => tokens.clone(),
        };

        let output = self.forward(&tokens, None, config.return_hidden_states)?;

        Ok(LogitsOutput {
            sequence_logits: if config.sequence {
                Some(output.sequence_logits)
            } else {
                None
            },
            embeddings: if config.return_embeddings {
                output.embeddings
            } else {
                None
            },
            hidden_states: output.hidden_states,
        })
    }
}
