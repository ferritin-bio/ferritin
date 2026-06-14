//! ESM3 multi-track input embedding (EncodeInputs).
//!
//! Embeds up to 8 input tracks into a single `(B, L, d_model)` tensor by summing
//! per-track contributions. All tracks are optional; missing tracks contribute zero.

use crate::esm3::models::esm3::ESM3Config;
use candle_core::{D, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

// ── EmbeddingBag ─────────────────────────────────────────────────────────────

/// Sum-mode embedding bag: looks up embeddings for multiple indices per position
/// and sums them, zeroing out the padding index.
pub struct EmbeddingBag {
    embed: nn::Embedding,
    padding_idx: u32,
}

impl EmbeddingBag {
    pub fn load(
        vb: VarBuilder,
        vocab_size: usize,
        embed_dim: usize,
        padding_idx: u32,
    ) -> Result<Self> {
        Ok(Self {
            embed: nn::embedding(vocab_size, embed_dim, vb)?,
            padding_idx,
        })
    }

    /// `indices`: `(*, K)` — K annotation IDs per position; `padding_idx` entries → zero.
    /// Returns `(*, embed_dim)` via sum over K.
    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        // Look up: (*, K) → (*, K, embed_dim)
        let embedded = self.embed.forward(indices)?;

        // Mask out padding: where index == padding_idx, contribution = 0
        let pad = self.padding_idx;
        let mask = indices
            .ne(pad)?
            .unsqueeze(D::Minus1)?
            .broadcast_as(embedded.shape())?
            .to_dtype(embedded.dtype())?;
        let masked = (embedded * mask)?;

        // Sum over K dimension (second-to-last)
        masked.sum(D::Minus2)
    }
}

// ── RBF encoding ─────────────────────────────────────────────────────────────

/// Radial Basis Function encoding of scalar values into `n_bins` features.
///
/// `values`: `(*)` float tensor in `[v_min, v_max]`.
/// Returns `(*, n_bins)`.
fn rbf(values: &Tensor, v_min: f64, v_max: f64, n_bins: usize) -> Result<Tensor> {
    let device = values.device();
    let dtype = values.dtype();

    // Evenly spaced centers in [v_min, v_max]
    let centers: Vec<f32> = (0..n_bins)
        .map(|i| (v_min + (v_max - v_min) * i as f64 / (n_bins - 1) as f64) as f32)
        .collect();
    let centers = Tensor::new(centers.as_slice(), device)?
        .to_dtype(dtype)?
        .reshape((1, 1, n_bins))?; // (1, 1, n_bins) for broadcasting

    let width = (v_max - v_min) / (n_bins - 1) as f64;
    let denom = 2.0 * width * width;

    let v = values.unsqueeze(D::Minus1)?; // (*, 1)
    let diff = v.broadcast_sub(&centers)?; // (*, n_bins)
    diff.sqr()?.affine(-1.0 / denom, 0.0)?.exp()
}

// ── EncodeInputs ──────────────────────────────────────────────────────────────

pub struct EncodeInputs {
    d_model: usize,
    // Sequence track
    sequence_embed: nn::Embedding,
    // pLDDT tracks (projected from 16-bin RBF)
    plddt_projection: nn::Linear,
    structure_per_res_plddt_projection: nn::Linear,
    // Structure token track
    structure_tokens_embed: nn::Embedding,
    // Secondary structure and SASA
    ss8_embed: nn::Embedding,
    sasa_embed: nn::Embedding,
    // Function annotation tracks (8 separate embeddings concatenated)
    function_embeds: Vec<nn::Embedding>,
    n_function_tracks: usize,
    // Residue (InterPro) annotation track (EmbeddingBag, sum mode)
    residue_embed: EmbeddingBag,
}

impl EncodeInputs {
    pub fn load(vb: VarBuilder, config: &ESM3Config) -> Result<Self> {
        let d = config.d_model;
        let n_rbf = config.n_rbf_bins;

        let sequence_embed = nn::embedding(config.d_sequence_vocab, d, vb.pp("sequence_embed"))?;

        let plddt_projection = nn::linear_no_bias(n_rbf, d, vb.pp("plddt_projection"))?;
        let structure_per_res_plddt_projection =
            nn::linear_no_bias(n_rbf, d, vb.pp("structure_per_res_plddt_projection"))?;

        // Structure vocab: 4096 codes + 5 special tokens
        let structure_tokens_embed = nn::embedding(
            config.d_structure_vocab + 5,
            d,
            vb.pp("structure_tokens_embed"),
        )?;

        let ss8_embed = nn::embedding(config.d_ss8_vocab, d, vb.pp("ss8_embed"))?;
        let sasa_embed = nn::embedding(config.d_sasa_vocab, d, vb.pp("sasa_embed"))?;

        // 8 function-track embeddings, each produces d_model // 8 features
        let func_dim = d / config.n_function_tracks;
        let mut function_embeds = Vec::with_capacity(config.n_function_tracks);
        for i in 0..config.n_function_tracks {
            function_embeds.push(nn::embedding(
                config.d_function_vocab,
                func_dim,
                vb.pp(format!("function_embed.{}", i)),
            )?);
        }

        let residue_embed = EmbeddingBag::load(
            vb.pp("residue_embed"),
            config.d_residue_vocab,
            d,
            0, // padding_idx
        )?;

        Ok(Self {
            d_model: d,
            sequence_embed,
            plddt_projection,
            structure_per_res_plddt_projection,
            structure_tokens_embed,
            ss8_embed,
            sasa_embed,
            function_embeds,
            n_function_tracks: config.n_function_tracks,
            residue_embed,
        })
    }

    /// Embed all input tracks and sum them.
    ///
    /// All arguments are optional; present tracks contribute their embedding,
    /// absent tracks contribute zero.
    ///
    /// - `sequence_tokens`:           `(B, L)` u32 sequence token IDs.
    /// - `structure_tokens`:          `(B, L)` u32 structure (VQ-VAE) tokens.
    /// - `ss8_tokens`:                `(B, L)` u32 secondary-structure tokens.
    /// - `sasa_tokens`:               `(B, L)` u32 SASA-bin tokens.
    /// - `function_tokens`:           `(B, L, n_tracks)` u32 function annotation tokens.
    /// - `residue_annotation_tokens`: `(B, L, K)` u32 InterPro annotation IDs.
    /// - `average_plddt`:             `(B, L)` f32 average per-structure pLDDT in [0,1].
    /// - `per_res_plddt`:             `(B, L)` f32 per-residue pLDDT in [0,1].
    ///
    /// Returns `(B, L, d_model)`.
    pub fn forward(
        &self,
        sequence_tokens: Option<&Tensor>,
        structure_tokens: Option<&Tensor>,
        ss8_tokens: Option<&Tensor>,
        sasa_tokens: Option<&Tensor>,
        function_tokens: Option<&Tensor>,
        residue_annotation_tokens: Option<&Tensor>,
        average_plddt: Option<&Tensor>,
        per_res_plddt: Option<&Tensor>,
    ) -> Result<Tensor> {
        let n_rbf = 16usize;

        // Accumulate embeddings into x; first present track initialises x.
        let mut x: Option<Tensor> = None;
        let mut add = |t: Tensor| -> Result<()> {
            x = Some(match x.take() {
                None => t,
                Some(acc) => acc.add(&t)?,
            });
            Ok(())
        };

        if let Some(seq) = sequence_tokens {
            add(self.sequence_embed.forward(seq)?)?;
        }

        if let Some(st) = structure_tokens {
            add(self.structure_tokens_embed.forward(st)?)?;
        }

        if let Some(ss8) = ss8_tokens {
            add(self.ss8_embed.forward(ss8)?)?;
        }

        if let Some(sasa) = sasa_tokens {
            add(self.sasa_embed.forward(sasa)?)?;
        }

        if let Some(plddt) = average_plddt {
            let enc = rbf(plddt, 0.0, 1.0, n_rbf)?;
            add(self.plddt_projection.forward(&enc)?)?;
        }

        if let Some(per_res) = per_res_plddt {
            let enc = rbf(per_res, 0.0, 1.0, n_rbf)?;
            add(self.structure_per_res_plddt_projection.forward(&enc)?)?;
        }

        if let Some(func) = function_tokens {
            // func: (B, L, n_tracks); each track uses its own embedding
            let mut parts: Vec<Tensor> = Vec::with_capacity(self.n_function_tracks);
            for i in 0..self.n_function_tracks {
                let track = func.narrow(D::Minus1, i, 1)?.squeeze(D::Minus1)?; // (B, L)
                parts.push(self.function_embeds[i].forward(&track)?); // (B, L, func_dim)
            }
            let func_emb = Tensor::cat(&parts, D::Minus1)?; // (B, L, d_model)
            add(func_emb)?;
        }

        if let Some(res) = residue_annotation_tokens {
            add(self.residue_embed.forward(res)?)?;
        }

        // If no tracks provided, return zeros. In practice at least sequence_tokens is present.
        match x {
            Some(t) => Ok(t),
            None => candle_core::bail!("EncodeInputs: at least one input track must be provided"),
        }
    }
}
