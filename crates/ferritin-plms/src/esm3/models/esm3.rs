//! ESM3 multimodal protein language model.

use crate::esm3::layers::encode_inputs::EncodeInputs;
use crate::esm3::layers::output_heads::{ESM3Output, OutputHeads};
use crate::esm3::layers::transformer_stack::TransformerStack;
use crate::esm3::utils::affine3d::Affine3D;
use crate::esm3::utils::constants::{
    INTERPRO_PAD_TOKEN, RESIDUE_PAD_TOKEN, SASA_PAD_TOKEN, SEQUENCE_BOS_TOKEN,
    SEQUENCE_CHAINBREAK_TOKEN, SEQUENCE_EOS_TOKEN, SEQUENCE_MASK_TOKEN, SEQUENCE_PAD_TOKEN,
    SS8_PAD_TOKEN, STRUCTURE_BOS_TOKEN, STRUCTURE_CHAINBREAK_TOKEN, STRUCTURE_EOS_TOKEN,
    STRUCTURE_MASK_TOKEN, STRUCTURE_PAD_TOKEN,
};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

// ── ESM3Config ────────────────────────────────────────────────────────────────

/// Configuration for ESM3.
#[derive(Debug, Clone)]
pub struct ESM3Config {
    // ── Transformer ──
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    /// Number of transformer layers that use geometric (affine3d) attention.
    pub n_layers_geom: usize,
    /// Number of heads for geometric attention (v_heads).
    pub v_head_transformer: usize,
    /// FFN expansion ratio.
    pub expansion_ratio: f64,
    /// Whether to scale residual connections by sqrt(n_layers / 36).
    pub scale_residue: bool,
    /// Whether to mask and zero frameless positions in geometric attention.
    pub mask_and_zero_frameless: bool,
    /// Whether to apply per-head LayerNorm to Q and K.
    pub qk_layernorm: bool,
    /// Whether to use bias in linear layers.
    pub bias: bool,
    // ── Vocab sizes ──
    pub d_sequence_vocab: usize,
    pub d_structure_vocab: usize,
    /// SS8 secondary-structure vocabulary size (8 classes + 3 special = 11).
    pub d_ss8_vocab: usize,
    /// SASA vocabulary size (16 bins + 3 special = 19).
    pub d_sasa_vocab: usize,
    /// Number of function annotation tracks (8 in ESM3).
    pub n_function_tracks: usize,
    /// Vocabulary size per function track.
    pub d_function_vocab: usize,
    /// InterPro residue annotation vocabulary size.
    pub d_residue_vocab: usize,
    /// Number of RBF bins for pLDDT encoding (16).
    pub n_rbf_bins: usize,
}

impl ESM3Config {
    /// Config for `esm3-sm-open-v1` (1.4B parameters).
    pub fn sm_open() -> Self {
        Self {
            d_model: 1536,
            n_heads: 24,
            n_layers: 48,
            n_layers_geom: 1,
            v_head_transformer: 256,
            expansion_ratio: 8.0 / 3.0,
            scale_residue: true,
            mask_and_zero_frameless: false,
            qk_layernorm: true,
            bias: false,
            d_sequence_vocab: 64,
            d_structure_vocab: 4096,
            d_ss8_vocab: 11,  // 8 + 3 special
            d_sasa_vocab: 19, // 16 bins + 3 special
            n_function_tracks: 8,
            d_function_vocab: 260,
            d_residue_vocab: 1478,
            n_rbf_bins: 16,
        }
    }

    /// Residual scaling factor: `sqrt(n_layers / 36)` when `scale_residue` is true.
    pub fn residue_scaling_factor(&self) -> f64 {
        if self.scale_residue {
            (self.n_layers as f64 / 36.0).sqrt()
        } else {
            1.0
        }
    }
}

// ── ESM3 model ────────────────────────────────────────────────────────────────

/// ESM3 multimodal protein language model.
///
/// Takes multi-track inputs (sequence, structure, SS8, SASA, function, pLDDT),
/// encodes them through a geometric transformer, and predicts per-track logits.
pub struct ESM3 {
    pub config: ESM3Config,
    encode_inputs: EncodeInputs,
    transformer: TransformerStack,
    output_heads: OutputHeads,
}

impl ESM3 {
    pub fn load(vb: VarBuilder, config: ESM3Config) -> Result<Self> {
        let encode_inputs = EncodeInputs::load(vb.pp("encoder"), &config)?;
        let transformer = TransformerStack::load(vb.pp("transformer"), &config)?;
        let output_heads = OutputHeads::load(vb.pp("output_heads"), &config)?;
        Ok(Self {
            config,
            encode_inputs,
            transformer,
            output_heads,
        })
    }

    /// Every input track, with absent ones filled in the way upstream
    /// `ESM3.forward` fills them.
    ///
    /// This is the whole of ferritin-100.31. The port previously *skipped* an
    /// absent track — `if let Some(t) = track { add(embed(t)) }` — on the
    /// reasonable-sounding assumption that "no data" means "no contribution".
    /// Upstream instead materialises a full pad/mask tensor for every missing
    /// track and embeds it, so a sequence-only forward still sums eight
    /// contributions, not one. Six of them are constant across positions, but
    /// the structure track is not: it carries its own BOS/EOS tokens, distinct
    /// from mask.
    ///
    /// Measured against `esm3_parity.safetensors`, skipping the tracks put
    /// per-residue cosine similarity at 0.80-0.89 and left the BOS/EOS rows
    /// with roughly 15x the reference norm. Filling them takes every position
    /// to 1.000000.
    #[allow(clippy::too_many_arguments)]
    fn default_tracks(
        sequence_tokens: Option<&Tensor>,
        structure_tokens: Option<&Tensor>,
        ss8_tokens: Option<&Tensor>,
        sasa_tokens: Option<&Tensor>,
        function_tokens: Option<&Tensor>,
        residue_annotation_tokens: Option<&Tensor>,
        average_plddt: Option<&Tensor>,
        per_res_plddt: Option<&Tensor>,
    ) -> Result<DefaultTracks> {
        // Shape comes from whichever track the caller did supply; upstream
        // takes the first non-None the same way.
        let reference = [sequence_tokens, structure_tokens, ss8_tokens, sasa_tokens]
            .into_iter()
            .flatten()
            .next()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "ESM3::forward: at least one of the token tracks must be supplied".into(),
                )
            })?;
        let (batch, seq_len) = reference.dims2()?;
        let device = reference.device();

        let filled =
            |token: u32| -> Result<Tensor> { Tensor::full(token, (batch, seq_len), device) };
        let filled_3d = |token: u32, width: usize| -> Result<Tensor> {
            Tensor::full(token, (batch, seq_len, width), device)
        };
        let filled_f32 =
            |value: f32| -> Result<Tensor> { Tensor::full(value, (batch, seq_len), device) };

        let sequence_tokens = match sequence_tokens {
            Some(t) => t.clone(),
            None => filled(SEQUENCE_MASK_TOKEN)?,
        };

        // The structure track is the one that is not position-constant: start
        // from mask everywhere, then let the sequence track's special tokens
        // dictate the structure special tokens at those positions. Dropping
        // this is what wrecked the BOS/EOS rows.
        let structure_tokens = match structure_tokens {
            Some(t) => t.clone(),
            None => filled(STRUCTURE_MASK_TOKEN)?,
        };
        let structure_tokens = replace_where(
            &structure_tokens,
            &sequence_tokens,
            &[
                (SEQUENCE_BOS_TOKEN, STRUCTURE_BOS_TOKEN),
                (SEQUENCE_PAD_TOKEN, STRUCTURE_PAD_TOKEN),
                (SEQUENCE_EOS_TOKEN, STRUCTURE_EOS_TOKEN),
                (SEQUENCE_CHAINBREAK_TOKEN, STRUCTURE_CHAINBREAK_TOKEN),
            ],
        )?;

        Ok(DefaultTracks {
            sequence_tokens,
            structure_tokens,
            ss8_tokens: match ss8_tokens {
                Some(t) => t.clone(),
                None => filled(SS8_PAD_TOKEN)?,
            },
            sasa_tokens: match sasa_tokens {
                Some(t) => t.clone(),
                None => filled(SASA_PAD_TOKEN)?,
            },
            function_tokens: match function_tokens {
                Some(t) => t.clone(),
                None => filled_3d(INTERPRO_PAD_TOKEN, 8)?,
            },
            residue_annotation_tokens: match residue_annotation_tokens {
                Some(t) => t.clone(),
                None => filled_3d(RESIDUE_PAD_TOKEN, 16)?,
            },
            // Upstream's defaults are `average_plddt = 1`, `per_res_plddt = 0`
            // — not the same value, and not both zero.
            average_plddt: match average_plddt {
                Some(t) => t.clone(),
                None => filled_f32(1.0)?,
            },
            per_res_plddt: match per_res_plddt {
                Some(t) => t.clone(),
                None => filled_f32(0.0)?,
            },
        })
    }

    /// Forward pass through the full ESM3 model.
    ///
    /// All input tracks are optional, but an omitted track does **not**
    /// contribute zero: it is filled with its pad/mask token and embedded, the
    /// same way `ESM3.forward` does upstream. See `Self::default_tracks` (a
    /// private helper, so not linkable from here) — getting this wrong is not
    /// a small error (ferritin-100.31).
    ///
    /// At least `sequence_tokens` should be provided.
    ///
    /// - `structure_coords`: `(B, L, 3, 3)` backbone `(N, CA, C)` coordinates — used to
    ///   build per-residue affine frames for geometric attention.
    ///
    /// Returns `ESM3Output` with per-track logit distributions.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        sequence_tokens: Option<&candle_core::Tensor>,
        structure_tokens: Option<&candle_core::Tensor>,
        ss8_tokens: Option<&candle_core::Tensor>,
        sasa_tokens: Option<&candle_core::Tensor>,
        function_tokens: Option<&candle_core::Tensor>,
        residue_annotation_tokens: Option<&candle_core::Tensor>,
        average_plddt: Option<&candle_core::Tensor>,
        per_res_plddt: Option<&candle_core::Tensor>,
        sequence_id: Option<&candle_core::Tensor>,
        structure_coords: Option<&candle_core::Tensor>,
        chain_id: Option<&candle_core::Tensor>,
    ) -> Result<ESM3Output> {
        let tracks = Self::default_tracks(
            sequence_tokens,
            structure_tokens,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
            average_plddt,
            per_res_plddt,
        )?;

        // Embed all input tracks → (B, L, d_model)
        let x = self.encode_inputs.forward(
            Some(&tracks.sequence_tokens),
            Some(&tracks.structure_tokens),
            Some(&tracks.ss8_tokens),
            Some(&tracks.sasa_tokens),
            Some(&tracks.function_tokens),
            Some(&tracks.residue_annotation_tokens),
            Some(&tracks.average_plddt),
            Some(&tracks.per_res_plddt),
        )?;

        // Build per-residue affine frames from backbone coordinates (if provided)
        let affine_and_mask = structure_coords
            .map(Affine3D::build_affine3d_from_coordinates)
            .transpose()?;

        let (affine_ref, mask_ref);
        let (affine_opt, mask_opt) = match affine_and_mask {
            Some((ref aff, ref mask)) => {
                affine_ref = aff;
                mask_ref = mask;
                (Some(affine_ref), Some(mask_ref))
            }
            None => (None, None),
        };

        // Transformer stack
        let (post_norm, pre_norm) =
            self.transformer
                .forward(&x, sequence_id, affine_opt, mask_opt, chain_id)?;

        // Project to per-track logits
        let mut output = self.output_heads.forward(&post_norm)?;
        output.embeddings = Some(pre_norm);
        Ok(output)
    }
}

/// Every input track ESM3's encoder consumes, none of them optional.
///
/// Built by [`ESM3::default_tracks`]; exists so `forward` cannot accidentally
/// pass `None` for a track again.
struct DefaultTracks {
    sequence_tokens: Tensor,
    structure_tokens: Tensor,
    ss8_tokens: Tensor,
    sasa_tokens: Tensor,
    function_tokens: Tensor,
    residue_annotation_tokens: Tensor,
    average_plddt: Tensor,
    per_res_plddt: Tensor,
}

/// `tokens` with each `(when, then)` applied where `key` equals `when`.
///
/// candle has no `masked_fill`, so this is the `where_cond` spelling of
/// torch's chained `.masked_fill(key == when, then)`.
fn replace_where(tokens: &Tensor, key: &Tensor, rules: &[(u32, u32)]) -> Result<Tensor> {
    let mut out = tokens.clone();
    for &(when, then) in rules {
        let hit = key.eq(when)?;
        let replacement = Tensor::full(then, out.shape(), out.device())?.to_dtype(out.dtype())?;
        out = hit.where_cond(&replacement, &out)?;
    }
    Ok(out)
}
