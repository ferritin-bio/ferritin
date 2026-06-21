//! ESM3 output projection heads (OutputHeads).
//!
//! Projects the transformer hidden state to per-track logit distributions.

use crate::esm3::models::esm3::ESM3Config;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, LayerNormConfig, VarBuilder};

// ── Generic regression head ───────────────────────────────────────────────────

/// Linear(d_in) → GELU → LayerNorm → Linear(d_out).
pub struct RegressionHead {
    model: nn::Sequential,
}

impl RegressionHead {
    pub fn load(vb: VarBuilder, d_in: usize, d_out: usize) -> Result<Self> {
        let ln_conf = LayerNormConfig::from(1e-5);
        let model = nn::seq()
            .add(nn::linear(d_in, d_in, vb.pp("0"))?)
            .add(nn::Activation::Gelu)
            .add(nn::layer_norm(d_in, ln_conf, vb.pp("2"))?)
            .add(nn::linear(d_in, d_out, vb.pp("3"))?);
        Ok(Self { model })
    }
}

impl Module for RegressionHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.model.forward(x)
    }
}

// ── OutputHeads ───────────────────────────────────────────────────────────────

pub struct OutputHeads {
    sequence_head: RegressionHead,
    structure_head: RegressionHead,
    ss8_head: RegressionHead,
    sasa_head: RegressionHead,
    function_head: RegressionHead,
    residue_head: RegressionHead,
    n_function_tracks: usize,
    d_function_vocab: usize,
}

impl OutputHeads {
    pub fn load(vb: VarBuilder, config: &ESM3Config) -> Result<Self> {
        let d = config.d_model;
        Ok(Self {
            sequence_head: RegressionHead::load(
                vb.pp("sequence_head"),
                d,
                config.d_sequence_vocab,
            )?,
            structure_head: RegressionHead::load(
                vb.pp("structure_head"),
                d,
                config.d_structure_vocab,
            )?,
            ss8_head: RegressionHead::load(vb.pp("ss8_head"), d, config.d_ss8_vocab)?,
            sasa_head: RegressionHead::load(vb.pp("sasa_head"), d, config.d_sasa_vocab)?,
            function_head: RegressionHead::load(
                vb.pp("function_head"),
                d,
                config.n_function_tracks * config.d_function_vocab,
            )?,
            residue_head: RegressionHead::load(vb.pp("residue_head"), d, config.d_residue_vocab)?,
            n_function_tracks: config.n_function_tracks,
            d_function_vocab: config.d_function_vocab,
        })
    }

    /// Project hidden states to per-track logit distributions.
    ///
    /// `x`: `(B, L, d_model)` — post-norm transformer output.
    ///
    /// Returns `ESM3Output` with optional logit tensors.
    pub fn forward(&self, x: &Tensor) -> Result<ESM3Output> {
        let (b, l, _) = x.dims3()?;

        // Function logits: (B, L, n_tracks * d_func_vocab) → (B, L, n_tracks, d_func_vocab)
        let function_logits = self.function_head.forward(x)?.reshape((
            b,
            l,
            self.n_function_tracks,
            self.d_function_vocab,
        ))?;

        Ok(ESM3Output {
            sequence_logits: Some(self.sequence_head.forward(x)?),
            structure_logits: Some(self.structure_head.forward(x)?),
            secondary_structure_logits: Some(self.ss8_head.forward(x)?),
            sasa_logits: Some(self.sasa_head.forward(x)?),
            function_logits: Some(function_logits),
            residue_logits: Some(self.residue_head.forward(x)?),
            embeddings: None,
        })
    }
}

// ── ESM3Output ────────────────────────────────────────────────────────────────

/// Output of an ESM3 forward pass.
#[derive(Debug)]
pub struct ESM3Output {
    /// `(B, L, d_sequence_vocab)` — per-residue amino-acid logits.
    pub sequence_logits: Option<Tensor>,
    /// `(B, L, d_structure_vocab)` — per-residue structure-token logits.
    pub structure_logits: Option<Tensor>,
    /// `(B, L, d_ss8_vocab)` — secondary-structure logits.
    pub secondary_structure_logits: Option<Tensor>,
    /// `(B, L, d_sasa_vocab)` — SASA-bin logits.
    pub sasa_logits: Option<Tensor>,
    /// `(B, L, n_function_tracks, d_function_vocab)` — function annotation logits.
    pub function_logits: Option<Tensor>,
    /// `(B, L, d_residue_vocab)` — InterPro residue-annotation logits.
    pub residue_logits: Option<Tensor>,
    /// `(B, L, d_model)` — final hidden states (when requested).
    pub embeddings: Option<Tensor>,
}
