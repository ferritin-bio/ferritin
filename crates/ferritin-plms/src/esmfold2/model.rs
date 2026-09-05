//! ESMFold2Model — wires ESMC-6B backbone + structure-head layers.

use super::config::ESMFold2Config;
#[allow(unused_imports)]
use super::layers::{
    atom_encoder::AtomEncoder, // TODO: add atom_encoder field when AtomEncoder is complete
    confidence_head::{ConfidenceHead, bins_to_scalar},
    diffusion::DiffusionModule,
    folding_trunk::FoldingTrunk,
    lm_encoder::LMEncoder,
};
use super::output::ESMFold2Output;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

/// Full ESMFold2-Fast structure prediction model.
///
/// The ESMC-6B backbone (frozen) is loaded separately; this struct holds only
/// the structure-head components that are fine-tuned for structure prediction.
pub struct ESMFold2Model {
    #[allow(dead_code)]
    config: ESMFold2Config,
    lm_encoder: LMEncoder,
    folding_trunk: FoldingTrunk,
    diffusion: DiffusionModule,
    confidence_head: ConfidenceHead,
    device: Device,
}

impl ESMFold2Model {
    /// Load the ESMFold2-Fast structure head from a `VarBuilder`.
    ///
    /// The `VarBuilder` should be rooted at the structure-head safetensors
    /// (`biohub/ESMFold2-Fast model.safetensors`). The ESMC-6B backbone is
    /// loaded separately.
    pub fn load(vb: VarBuilder, config: ESMFold2Config) -> Result<Self> {
        let device = vb.device().clone();

        let lm_encoder = LMEncoder::load(
            vb.pp("lm_encoder"),
            config.lm_d_model,
            config.d_single,
            config.lm_encoder_n_layers,
        )?;

        let folding_trunk = FoldingTrunk::load(
            vb.pp("folding_trunk"),
            config.trunk_n_layers,
            config.d_single,
            config.d_pair,
            config.trunk_n_heads,
        )?;

        let diffusion = DiffusionModule::load(
            vb.pp("structure_head"),
            config.c_token,
            config.c_atom,
            config.d_single,
            config.d_pair,
            config.token_num_blocks,
            config.token_num_heads,
            config.fourier_dim,
        )?;

        let confidence_head = ConfidenceHead::load(
            vb.pp("confidence_head"),
            config.d_single,
            config.d_pair,
            config.num_plddt_bins,
            config.num_pae_bins,
            config.num_pde_bins,
            config.distogram_bins,
        )?;

        Ok(Self {
            config,
            lm_encoder,
            folding_trunk,
            diffusion,
            confidence_head,
            device,
        })
    }

    /// Run the full ESMFold2 structure-head forward pass.
    ///
    /// # Arguments
    /// * `hidden_states`    — ESMC-6B backbone output `[B, L, 2560]`
    /// * `residue_indices`  — residue positions `[B, L]`, any numeric dtype
    /// * `chain_ids`        — chain identifiers `[B, L]`, any numeric dtype
    /// * `num_loops`        — number of recycling iterations (typically 3)
    /// * `num_steps`        — diffusion denoising steps (14 fast / 50 quality)
    ///
    /// # Returns
    /// [`ESMFold2Output`] with coordinates, pLDDT, and optional PAE/distogram.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residue_indices: &Tensor,
        chain_ids: &Tensor,
        num_loops: usize,
        num_steps: usize,
    ) -> Result<ESMFold2Output> {
        let (b, l, _) = hidden_states.dims3()?;

        // LMEncoder: [B, L, 2560] → [B, L, d_single=384]
        let mut single = self.lm_encoder.forward(hidden_states)?;

        // Initialise pair from sequence metadata: [B, L, L, d_pair=256]
        let mut pair = self
            .folding_trunk
            .init_pair(&single, residue_indices, chain_ids)?;

        // Recycling loops through FoldingTrunk
        for _ in 0..num_loops {
            let (s, p) = self.folding_trunk.forward(&single, &pair)?;
            single = s;
            pair = p;
        }

        // Diffusion: [B, L, 3] token-level Cα coordinates
        let coords = self.diffusion.forward(&single, &pair, l, num_steps)?;

        // Confidence head: pLDDT, pAE, distogram
        let conf = self.confidence_head.forward(&single, &pair)?;

        // Convert pAE logits [B, N, N, 64] → scalar PAE [B, N, N] in Å (0..32)
        let pae = conf
            .pae_logits
            .map(|t| bins_to_scalar(&t, 0.0, 32.0))
            .transpose()?;

        // Stub ptm/iptm as zeros [B] — requires frame-aligned point error calc
        let ptm = Tensor::zeros(b, DType::F32, &self.device)?;
        let iptm = Tensor::zeros(b, DType::F32, &self.device)?;

        Ok(ESMFold2Output {
            sample_atom_coords: coords,
            plddt: conf.plddt,
            ptm,
            iptm,
            pae,
            distogram_logits: conf.distogram_logits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_model(device: &Device) -> ESMFold2Model {
        use super::super::config::ESMFold2Config;
        let cfg = ESMFold2Config {
            lm_d_model: 64,
            lm_encoder_n_layers: 1,
            d_single: 32,
            d_pair: 16,
            trunk_n_layers: 1,
            trunk_n_heads: 2,
            c_token: 32,
            c_atom: 16,
            token_num_blocks: 1,
            token_num_heads: 2,
            fourier_dim: 16,
            num_plddt_bins: 50,
            num_pae_bins: 64,
            num_pde_bins: 64,
            distogram_bins: 39,
            ..ESMFold2Config::fast()
        };
        let vb = VarBuilder::zeros(DType::F32, device);
        ESMFold2Model::load(vb, cfg).unwrap()
    }

    #[test]
    fn test_forward_output_shapes() {
        let device = Device::Cpu;
        let model = make_model(&device);

        let b = 1usize;
        let l = 8usize;
        let hidden = Tensor::zeros(&[b, l, 64], DType::F32, &device).unwrap();
        let res_idx: Vec<f32> = (0..l).map(|i| i as f32).collect();
        let residue_indices = Tensor::from_vec(res_idx, &[b, l], &device).unwrap();
        let chain_ids = Tensor::zeros(&[b, l], DType::F32, &device).unwrap();

        let out = model
            .forward(&hidden, &residue_indices, &chain_ids, 1, 2)
            .unwrap();

        assert_eq!(out.sample_atom_coords.dims(), &[b, l, 3]);
        assert_eq!(out.plddt.dims(), &[b, l]);
        assert_eq!(out.ptm.dims(), &[b]);
        assert_eq!(out.iptm.dims(), &[b]);
    }
}
