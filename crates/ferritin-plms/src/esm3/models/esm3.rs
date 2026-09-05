//! ESM3 multimodal protein language model.

use crate::esm3::layers::encode_inputs::EncodeInputs;
use crate::esm3::layers::output_heads::{ESM3Output, OutputHeads};
use crate::esm3::layers::transformer_stack::TransformerStack;
use crate::esm3::utils::affine3d::Affine3D;
use candle_core::Result;
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

    /// Forward pass through the full ESM3 model.
    ///
    /// All input tracks are optional; omitted tracks contribute zero to the
    /// initial embedding. At least `sequence_tokens` should be provided.
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
        // Embed all input tracks → (B, L, d_model)
        let x = self.encode_inputs.forward(
            sequence_tokens,
            structure_tokens,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
            average_plddt,
            per_res_plddt,
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
