//! ESM3 VQ-VAE structure token encoder and decoder (vqvae.py port).
//!
//! `StructureTokenEncoder` maps backbone coordinates to discrete structure tokens via a
//! 2-layer geometric mini-transformer + nearest-neighbour VQ codebook lookup.
//!
//! `StructureTokenDecoder` is stubbed (not needed for ESM3 inference).

use crate::esm3::layers::transformer_stack::TransformerStack;
use crate::esm3::models::esm3::ESM3Config;
use crate::esm3::utils::affine3d::Affine3D;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

// ── VQ-VAE config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VqVaeConfig {
    // Encoder mini-transformer
    pub enc_d_model: usize,  // 1024
    pub enc_n_heads: usize,  // 1
    pub enc_v_heads: usize,  // 128
    pub enc_n_layers: usize, // 2
    // Codebook
    pub d_codebook: usize, // 128 (projected dimension before VQ)
    pub n_codes: usize,    // 4096
    // Decoder (kept for future use; not loaded in MVP)
    pub dec_d_model: usize,  // 1280
    pub dec_n_heads: usize,  // 20
    pub dec_n_layers: usize, // 30
}

impl Default for VqVaeConfig {
    fn default() -> Self {
        Self {
            enc_d_model: 1024,
            enc_n_heads: 1,
            enc_v_heads: 128,
            enc_n_layers: 2,
            d_codebook: 128,
            n_codes: 4096,
            dec_d_model: 1280,
            dec_n_heads: 20,
            dec_n_layers: 30,
        }
    }
}

impl VqVaeConfig {
    /// Build an `ESM3Config` for the encoder TransformerStack.
    ///
    /// All layers use geometric attention; vocab sizes are unused and zeroed.
    fn encoder_esm3_config(&self) -> ESM3Config {
        ESM3Config {
            d_model: self.enc_d_model,
            n_heads: self.enc_n_heads,
            n_layers: self.enc_n_layers,
            n_layers_geom: self.enc_n_layers, // every layer uses geometric attention
            v_head_transformer: self.enc_v_heads,
            expansion_ratio: 8.0 / 3.0,
            scale_residue: false,
            mask_and_zero_frameless: true,
            qk_layernorm: true,
            bias: false,
            // Vocab/embedding sizes unused by TransformerStack
            d_sequence_vocab: 0,
            d_structure_vocab: self.n_codes,
            d_ss8_vocab: 0,
            d_sasa_vocab: 0,
            n_function_tracks: 0,
            d_function_vocab: 0,
            d_residue_vocab: 0,
            n_rbf_bins: 0,
        }
    }
}

// ── Inference VQ codebook ─────────────────────────────────────────────────────

/// Inference-only VQ codebook: loads the `(n_codes, d_codebook)` embedding table and
/// performs nearest-neighbour quantization via L2 distances.
struct VqCodebook {
    embeddings: Tensor, // (n_codes, d_codebook)
}

impl VqCodebook {
    pub fn load(vb: VarBuilder, n_codes: usize, d_codebook: usize) -> Result<Self> {
        let embeddings = vb.get((n_codes, d_codebook), "embeddings")?;
        Ok(Self { embeddings })
    }

    /// Quantize `z` to the nearest codebook entry.
    ///
    /// `z`: `(B, L, d_codebook)`.
    /// Returns `(B, L)` u32 structure token indices.
    pub fn quantize(&self, z: &Tensor) -> Result<Tensor> {
        let (b, l, d) = z.dims3()?;
        let z_flat = z.reshape((b * l, d))?; // (B*L, d)

        // ||z - e||^2 = ||z||^2 - 2*(z @ e^T) + ||e||^2
        let z_sq = z_flat.sqr()?.sum_keepdim(1)?; // (B*L, 1)
        let z_et = z_flat.matmul(&self.embeddings.transpose(0, 1)?)?; // (B*L, n_codes)
        let e_sq = self.embeddings.sqr()?.sum_keepdim(1)?.transpose(0, 1)?; // (1, n_codes)

        // distances: (B*L, n_codes)
        let distances = z_sq
            .broadcast_sub(&z_et.affine(2.0, 0.0)?)?
            .broadcast_add(&e_sq)?;

        let indices = distances.argmin(1)?; // (B*L,) u32
        indices.reshape((b, l))
    }
}

// ── StructureTokenEncoder ─────────────────────────────────────────────────────

/// Encodes backbone coordinates into discrete structure tokens via a 2-layer geometric
/// mini-transformer followed by nearest-neighbour VQ codebook lookup.
///
/// Weight layout (in encoder checkpoint):
/// - `encoder.blocks.*`   — transformer
/// - `pre_vq_proj.weight` — `(d_codebook, enc_d_model)`
/// - `codebook.embeddings`— `(n_codes, d_codebook)`
pub struct StructureTokenEncoder {
    transformer: TransformerStack,
    pre_vq_proj: nn::Linear,
    codebook: VqCodebook,
    config: VqVaeConfig,
}

impl StructureTokenEncoder {
    pub fn load(vb: VarBuilder, config: VqVaeConfig) -> Result<Self> {
        let enc_cfg = config.encoder_esm3_config();
        let transformer = TransformerStack::load(vb.pp("encoder"), &enc_cfg)?;
        let pre_vq_proj =
            nn::linear_no_bias(config.enc_d_model, config.d_codebook, vb.pp("pre_vq_proj"))?;
        let codebook = VqCodebook::load(vb.pp("codebook"), config.n_codes, config.d_codebook)?;
        Ok(Self {
            transformer,
            pre_vq_proj,
            codebook,
            config,
        })
    }

    /// Encode backbone coordinates into structure tokens.
    ///
    /// - `coords`:      `(B, L, 3, 3)` backbone `(N, CA, C)` positions.
    /// - `sequence_id`: optional `(B, L)` bin-packing IDs.
    /// - `chain_id`:    optional `(B, L)` chain IDs.
    ///
    /// Returns `(B, L)` u32 structure token indices.
    pub fn encode(
        &self,
        coords: &Tensor,
        sequence_id: Option<&Tensor>,
        chain_id: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = coords.dim(0)?;
        let l = coords.dim(1)?;
        let device = coords.device();
        let dtype = coords.dtype();

        let (affine, affine_mask) = Affine3D::build_affine3d_from_coordinates(coords)?;

        // Initial hidden state: zeros (all structural info enters through geometric attention)
        let x = Tensor::zeros((b, l, self.config.enc_d_model), dtype, device)?;

        let (x, _pre_norm) = self.transformer.forward(
            &x,
            sequence_id,
            Some(&affine),
            Some(&affine_mask),
            chain_id,
        )?;

        let z = self.pre_vq_proj.forward(&x)?;
        self.codebook.quantize(&z)
    }
}

// ── StructureTokenDecoder (stub) ──────────────────────────────────────────────

/// Stub for the structure token decoder. Not needed for ESM3 inference.
///
/// The decoder (d_model=1280, n_heads=20, n_layers=30) maps structure tokens back to
/// coordinates, but ESM3 inference only needs the encoder.
pub struct StructureTokenDecoder;

impl StructureTokenDecoder {
    pub fn stub() -> Self {
        Self
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_vq_codebook_quantize_shape() -> Result<()> {
        let device = &Device::Cpu;
        let n_codes = 16usize;
        let d = 8usize;
        let b = 2usize;
        let l = 5usize;

        let embeddings = Tensor::randn(0f32, 1f32, (n_codes, d), device)?;
        let codebook = VqCodebook { embeddings };

        let z = Tensor::randn(0f32, 1f32, (b, l, d), device)?;
        let tokens = codebook.quantize(&z)?;

        assert_eq!(tokens.shape().dims(), &[b, l]);
        Ok(())
    }

    #[test]
    fn test_vq_codebook_nearest_neighbour() -> Result<()> {
        let device = &Device::Cpu;
        // Two codebook entries: [1,0] and [-1,0]
        let embeddings = Tensor::new(&[[1f32, 0.], [-1., 0.]], device)?;
        let codebook = VqCodebook { embeddings };

        // z close to code 0 ([1,0]): should map to index 0
        let z = Tensor::new(&[[[0.9f32, 0.1]]], device)?; // (1,1,2)
        let tokens = codebook.quantize(&z)?;
        assert_eq!(tokens.to_vec2::<u32>()?, vec![vec![0u32]]);

        // z close to code 1 ([-1,0]): should map to index 1
        let z = Tensor::new(&[[[-0.8f32, 0.1]]], device)?;
        let tokens = codebook.quantize(&z)?;
        assert_eq!(tokens.to_vec2::<u32>()?, vec![vec![1u32]]);
        Ok(())
    }

    #[test]
    fn test_vqvae_config_encoder_esm3_config() {
        let cfg = VqVaeConfig::default();
        let enc = cfg.encoder_esm3_config();
        assert_eq!(enc.d_model, 1024);
        assert_eq!(enc.n_heads, 1);
        assert_eq!(enc.n_layers, 2);
        assert_eq!(
            enc.n_layers_geom, 2,
            "all layers should use geometric attention"
        );
        assert_eq!(enc.v_head_transformer, 128);
        assert!(!enc.scale_residue);
        assert!(enc.mask_and_zero_frameless);
    }
}
