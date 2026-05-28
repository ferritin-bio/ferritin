//! ESMFold2Model — wires ESMC-6B backbone + structure-head layers.

use super::config::ESMFold2Config;
#[allow(unused_imports)]
use super::layers::{
    atom_encoder::AtomEncoder, // TODO: add atom_encoder field when AtomEncoder is complete
    confidence_head::ConfidenceHead,
    diffusion::DiffusionModule,
    folding_trunk::FoldingTrunk,
    lm_encoder::LMEncoder,
};
use candle_core::{Device, Result};
use candle_nn::VarBuilder;

/// Full ESMFold2-Fast structure prediction model.
///
/// The ESMC-6B backbone (frozen) is loaded separately; this struct holds only
/// the structure-head components that are fine-tuned for structure prediction.
pub struct ESMFold2Model {
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
        )?;

        let diffusion =
            DiffusionModule::load(vb.pp("structure_head"), config.c_token, config.c_atom)?;

        let confidence_head = ConfidenceHead::load(
            vb.pp("confidence_head"),
            config.num_plddt_bins,
            config.num_pae_bins,
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
}
