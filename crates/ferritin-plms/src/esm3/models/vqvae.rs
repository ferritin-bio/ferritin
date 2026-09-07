//! ESM3 VQ-VAE structure token encoder and decoder (vqvae.py port).
//!
//! `StructureTokenEncoder` maps backbone coordinates to discrete structure
//! tokens. It is a *local* encoder: for every residue it runs a 2-layer
//! geometric transformer over that residue's 16 nearest neighbours in space,
//! then quantizes the query node's output against a 4096-entry VQ codebook
//! (ferritin-100.22).
//!
//! `StructureTokenDecoder` is stubbed (not needed for ESM3 inference).

use crate::esm3::layers::relative_position::RelativePositionEmbedding;
use crate::esm3::layers::transformer_stack::TransformerStack;
use crate::esm3::models::esm3::ESM3Config;
use crate::esm3::utils::affine3d::Affine3D;
use candle_core::{D, DType, Module, Result, Tensor};
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
            // 4, not the main model's 8/3: with the 256-rounding correction
            // this gives an FFN hidden width of exactly 4096, matching the
            // checkpoint's ffn.1.weight [8192, 1024] (gated, so 2x4096) and
            // ffn.3.weight [1024, 4096]. The 8/3 used here before produced
            // 2816 and failed to load (ferritin-100.22).
            expansion_ratio: 4.0,
            scale_residue: false,
            mask_and_zero_frameless: true,
            qk_layernorm: true,
            // esm3_structure_encoder_v0.pth stores geom_attn with bias terms
            // (s_norm.bias, proj.bias, out_proj.bias), unlike the main model
            // (ferritin-100.21).
            bias: true,
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

// ── k-nearest-neighbour graph ─────────────────────────────────────────────────

/// Neighbours per residue in the structure encoder's local graph.
pub const KNN: usize = 16;

/// Distance used to push invalid residues to the end of every neighbour list.
///
/// Finite rather than infinite: these entries are still gathered and run
/// through the transformer (they are masked there), and an infinity would
/// propagate NaN through the frame arithmetic instead of being ignored.
const FAR: f64 = 1e9;

/// Build the k-nearest-neighbour edge list over CA positions.
///
/// * `ca`    — `(B, L, 3)` CA coordinates.
/// * `valid` — `(B, L)` 1 where the residue has a usable backbone frame.
///
/// Returns `(B, L, KNN)` `u32` neighbour indices, sorted by increasing
/// distance. Because a residue is at distance zero from itself, **index 0 of
/// each row is the residue itself** — the encoder relies on this to pick the
/// query node back out after the transformer.
///
/// Residues with no frame are pushed to the end of every list rather than
/// removed, so the edge tensor stays rectangular.
fn knn_edges(ca: &Tensor, valid: &Tensor, knn: usize) -> Result<Tensor> {
    let (b, l, _) = ca.dims3()?;
    let knn = knn.min(l);

    // Squared distances via ||a||^2 - 2 a·b + ||b||^2.
    let sq = ca.sqr()?.sum_keepdim(D::Minus1)?; // (B, L, 1)
    let cross = ca.matmul(&ca.transpose(1, 2)?)?; // (B, L, L)
    let d2 = (sq.broadcast_add(&sq.transpose(1, 2)?)? - cross.affine(2.0, 0.0)?)?;

    // Invalid residues become unreachable columns.
    let invalid = valid
        .to_dtype(d2.dtype())?
        .affine(-1.0, 1.0)? // 1 - valid
        .reshape((b, 1, l))?
        .broadcast_as((b, l, l))?;
    let d2 = (d2 + invalid.affine(FAR, 0.0)?)?;

    let order = d2.arg_sort_last_dim(true)?; // ascending
    order.narrow(D::Minus1, 0, knn)?.contiguous()
}

/// Gather one value per edge.
///
/// `src` is `(B, L, D)` and `edges` is `(B, L, K)`; the result is
/// `(B, L, K, D)` — for each residue, the `D`-vectors of its `K` neighbours.
fn node_gather(src: &Tensor, edges: &Tensor) -> Result<Tensor> {
    let (b, l, d) = src.dims3()?;
    let k = edges.dim(D::Minus1)?;
    // gather along the residue axis wants an index of the same rank as `src`,
    // so flatten the (L, K) grid into one axis and widen it across D.
    let idx = edges
        .reshape((b, l * k, 1))?
        .broadcast_as((b, l * k, d))?
        .contiguous()?;
    src.contiguous()?.gather(&idx, 1)?.reshape((b, l, k, d))
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
    relative_positional_embedding: RelativePositionEmbedding,
    pre_vq_proj: nn::Linear,
    codebook: VqCodebook,
    config: VqVaeConfig,
}

impl StructureTokenEncoder {
    /// Load from `esm3_structure_encoder_v0.pth`.
    ///
    /// Every prefix here was read off the real checkpoint rather than inferred
    /// from the reference source: it roots at `transformer` (not `encoder`),
    /// and `pre_vq_proj` carries a bias. Both were wrong before, and both
    /// would have failed loudly — unlike the missing relative position
    /// embedding, which simply left the encoder computing from zeros.
    pub fn load(vb: VarBuilder, config: VqVaeConfig) -> Result<Self> {
        let enc_cfg = config.encoder_esm3_config();
        let transformer = TransformerStack::load_geometric_encoder(vb.pp("transformer"), &enc_cfg)?;
        let relative_positional_embedding = RelativePositionEmbedding::load(
            vb.pp("relative_positional_embedding"),
            config.enc_d_model,
        )?;
        let pre_vq_proj = nn::linear(config.enc_d_model, config.d_codebook, vb.pp("pre_vq_proj"))?;
        let codebook = VqCodebook::load(vb.pp("codebook"), config.n_codes, config.d_codebook)?;
        Ok(Self {
            transformer,
            relative_positional_embedding,
            pre_vq_proj,
            codebook,
            config,
        })
    }

    /// Encode backbone coordinates into structure tokens.
    ///
    /// * `coords` — `(B, L, 3, 3)` backbone `(N, CA, C)` positions.
    /// * `sequence_id` — optional `(B, L)` bin-packing ids.
    ///
    /// Returns `(B, L)` `u32` structure token indices.
    ///
    /// # Why this is not a sequence transformer
    ///
    /// This encoder is **local**. It does not run the transformer over the
    /// chain; it runs it over each residue's 16 nearest neighbours in space,
    /// as `(B*L, 16, d_model)`. The earlier port ran a full-sequence pass from
    /// a zero hidden state, which is a different computation entirely — the
    /// weights would load (once the prefixes were right) and the output would
    /// be quietly meaningless.
    pub fn encode(&self, coords: &Tensor, sequence_id: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _, _) = coords.dims4()?;
        let device = coords.device();
        let dtype = coords.dtype();

        let (affine, affine_mask) = Affine3D::build_affine3d_from_coordinates(coords)?;

        // Neighbourhoods, in distance order: column 0 is the residue itself.
        let ca = coords.narrow(D::Minus2, 1, 1)?.squeeze(D::Minus2)?; // (B, L, 3)
        let edges = knn_edges(&ca, &affine_mask, KNN)?; // (B, L, K)
        let k = edges.dim(D::Minus1)?;

        // Gather each neighbourhood's frames and masks, flattening the residue
        // axis into the batch so the transformer sees B*L sequences of K.
        let rot =
            node_gather(&affine.rot.reshape((b, l, 9))?, &edges)?.reshape((b * l, k, 3, 3))?;
        let trans = node_gather(&affine.trans, &edges)?.reshape((b * l, k, 3))?;
        let knn_affine = Affine3D::new(rot, trans);

        let knn_affine_mask =
            node_gather(&affine_mask.to_dtype(dtype)?.unsqueeze(D::Minus1)?, &edges)?
                .reshape((b * l, k))?
                .to_dtype(affine_mask.dtype())?;

        let knn_sequence_id = match sequence_id {
            Some(sid) => node_gather(&sid.to_dtype(dtype)?.unsqueeze(D::Minus1)?, &edges)?
                .reshape((b * l, k))?
                .to_dtype(sid.dtype())?,
            None => Tensor::zeros((b * l, k), DType::U32, device)?,
        };
        let chain_id = Tensor::zeros((b * l, k), DType::U32, device)?;

        // The initial hidden state is the sequence offset to each neighbour —
        // there is no token embedding in this encoder.
        let res_idxs = edges.reshape((b * l, k))?;
        let centre = res_idxs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let z = self
            .relative_positional_embedding
            .forward(&centre, &res_idxs)?
            .to_dtype(dtype)?;

        let (z, _pre_norm) = self.transformer.forward(
            &z,
            Some(&knn_sequence_id),
            Some(&knn_affine),
            Some(&knn_affine_mask),
            Some(&chain_id),
        )?;

        // Take the query node back out: neighbours are distance-sorted, so it
        // is column 0.
        let z = z
            .reshape((b, l, k, self.config.enc_d_model))?
            .narrow(2, 0, 1)?
            .squeeze(2)?;

        let z = self.pre_vq_proj.forward(&z)?;
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

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use candle_core::Device;

    /// Ideal alpha-helix backbone: rise 1.5 A, 100 degrees per residue.
    ///
    /// Real secondary structure rather than a straight line, so the geometric
    /// attention has something to distinguish.
    fn helix(l: usize) -> Result<Tensor> {
        let (r, rise, turn) = (2.3f64, 1.5f64, 100f64.to_radians());
        let mut v: Vec<f32> = Vec::with_capacity(l * 9);
        for i in 0..l {
            for (k, off) in [(-1.0f64), 0.0, 1.0].iter().enumerate() {
                let _ = k;
                let t = (i as f64 + off * 0.35) * turn;
                let z = (i as f64 + off * 0.35) * rise;
                v.extend_from_slice(&[(r * t.cos()) as f32, (r * t.sin()) as f32, z as f32]);
            }
        }
        Tensor::from_vec(v, (1, l, 3, 3), &Device::Cpu)
    }

    /// Column 0 of every neighbour list is the residue itself.
    ///
    /// The encoder pulls the query node back out by taking index 0 after the
    /// transformer, so if this ordering broke, every residue would silently be
    /// tokenized as one of its neighbours.
    #[test]
    fn test_first_neighbour_is_the_residue_itself() -> Result<()> {
        let coords = helix(12)?;
        let ca = coords.narrow(D::Minus2, 1, 1)?.squeeze(D::Minus2)?;
        let valid = Tensor::ones((1, 12), DType::U8, &Device::Cpu)?;
        let edges = knn_edges(&ca, &valid, KNN)?;
        let first = edges
            .narrow(D::Minus1, 0, 1)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        assert_eq!(
            first,
            (0..12).collect::<Vec<u32>>(),
            "a residue is at distance zero from itself, so it must sort first"
        );
        Ok(())
    }

    /// Neighbour lists are ordered by increasing distance.
    #[test]
    fn test_neighbours_are_distance_sorted() -> Result<()> {
        let coords = helix(20)?;
        let ca = coords.narrow(D::Minus2, 1, 1)?.squeeze(D::Minus2)?;
        let valid = Tensor::ones((1, 20), DType::U8, &Device::Cpu)?;
        let edges = knn_edges(&ca, &valid, KNN)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let ca_v = ca.flatten_all()?.to_vec1::<f32>()?;
        let k = KNN.min(20);
        for i in 0..20 {
            let mut prev = -1.0f32;
            for j in 0..k {
                let n = edges[i * k + j] as usize;
                let d: f32 = (0..3)
                    .map(|c| (ca_v[i * 3 + c] - ca_v[n * 3 + c]).powi(2))
                    .sum();
                assert!(
                    d >= prev - 1e-4,
                    "neighbour {j} of residue {i} is out of order"
                );
                prev = d;
            }
        }
        Ok(())
    }

    /// A residue with no valid backbone frame is pushed to the end of every
    /// neighbour list rather than being chosen as a near neighbour.
    #[test]
    fn test_invalid_residues_are_not_selected_as_neighbours() -> Result<()> {
        let coords = helix(20)?;
        let ca = coords.narrow(D::Minus2, 1, 1)?.squeeze(D::Minus2)?;
        // Mark residue 1 invalid; it is spatially adjacent to residue 0.
        let mut mask = vec![1u8; 20];
        mask[1] = 0;
        let valid = Tensor::from_vec(mask, (1, 20), &Device::Cpu)?;
        let edges = knn_edges(&ca, &valid, KNN)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let k = KNN.min(20);
        // Residue 0's nearest few neighbours should skip the invalid residue 1.
        assert!(
            !edges[0..k / 2].contains(&1),
            "an invalid residue must not rank as a near neighbour: {:?}",
            &edges[0..k]
        );
        Ok(())
    }

    #[test]
    fn test_node_gather_selects_the_named_rows() -> Result<()> {
        // (1, 4, 2): row i is [i, i*10].
        let src = Tensor::from_vec(
            vec![0f32, 0., 1., 10., 2., 20., 3., 30.],
            (1, 4, 2),
            &Device::Cpu,
        )?;
        // One row per residue: 4 residues, 2 neighbours each.
        let edges = Tensor::from_vec(vec![2u32, 0, 3, 1, 0, 3, 1, 2], (1, 4, 2), &Device::Cpu)?;
        let out = node_gather(&src, &edges)?;
        assert_eq!(out.dims(), &[1, 4, 2, 2]);
        assert_eq!(
            out.flatten_all()?.to_vec1::<f32>()?,
            vec![
                2., 20., 0., 0., // residue 0 gathers rows 2 and 0
                3., 30., 1., 10., // residue 1 gathers rows 3 and 1
                0., 0., 3., 30., // residue 2 gathers rows 0 and 3
                1., 10., 2., 20., // residue 3 gathers rows 1 and 2
            ]
        );
        Ok(())
    }
}
