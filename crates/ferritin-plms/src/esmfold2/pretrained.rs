//! ESMFold2 pretrained model loading.
//!
//! Downloads the structure-head weights from HuggingFace and wires them with
//! the separately-loaded ESMC-6B backbone for end-to-end structure prediction.
//!
//! ## Weight layout
//!
//! `biohub/ESMFold2-Fast` `model.safetensors` (~755 MB) contains the structure
//! head only. The ESMC-6B backbone (~12 GB) must be loaded from a second repo
//! (`biohub/ESMC-6B`).
//!
//! | Weight prefix          | Component                                 |
//! |------------------------|-------------------------------------------|
//! | `lm_encoder.*`         | LM adapter (4 blocks, 2560 → d_single=384) |
//! | `folding_trunk.*`      | 24-layer Pairformer trunk                 |
//! | `inputs.atom_encoder.*`| Atom encoder (3 blocks, d_atom=128→768)   |
//! | `msa_encoder.*`        | MSA encoder (disabled in Fast)            |
//! | `structure_head.*`     | Diffusion module (token 12-block + atom 3-block) |
//! | `confidence_head.*`    | pLDDT / pAE / pDE / distogram heads       |
//!
//! The `from_pretrained` loader downloads both safetensors files and builds
//! the `ESMFold2Model`. The ESMC-6B backbone is loaded via `ESMCRunner`
//! (already implemented in `esmc::pretrained`).

use super::config::ESMFold2Config;
use super::model::ESMFold2Model;
use super::output::ESMFold2Output;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::HFClientSync;

const ESMFOLD2_DTYPE: DType = DType::F32;

/// Available ESMFold2 model variants hosted on HuggingFace.
pub enum ESMFold2Models {
    /// ESMFold2-Fast — single-sequence only, no MSA, optimized for speed.
    /// Use `num_sampling_steps = 50` for quality, `14` for speed.
    Fast,
    /// ESMFold2 (Full) — optional MSA conditioning for higher accuracy.
    Full,
}

impl ESMFold2Models {
    /// Returns `(hf_repo_id, config)` for this variant.
    pub fn model_info(&self) -> (&'static str, ESMFold2Config) {
        match self {
            Self::Fast => ("biohub/ESMFold2-Fast", ESMFold2Config::fast()),
            Self::Full => ("biohub/ESMFold2", ESMFold2Config::fast()), // TODO: add full config when MSA encoder is implemented
        }
    }
}

/// Wraps a loaded ESMFold2 model for all-atom structure prediction inference.
pub struct ESMFold2Runner {
    model: ESMFold2Model,
    config: ESMFold2Config,
    device: Device,
}

impl ESMFold2Runner {
    /// Download weights from HuggingFace and load the ESMFold2 structure head.
    ///
    /// This downloads `model.safetensors` from `biohub/ESMFold2-Fast` (~755 MB)
    /// and builds the structure-head via `ESMFold2Model::load`. The ESMC-6B
    /// backbone is not loaded here — it will be wired in once the layer
    /// implementations are complete.
    ///
    /// # Note
    /// `fold_protein` currently returns a stub error until the forward-pass
    /// implementations are filled in for each layer.
    pub fn from_pretrained(model_variant: ESMFold2Models, device: Device) -> Result<Self> {
        let (repo_id, config) = model_variant.model_info();

        eprintln!("ESMFold2Runner: downloading structure head from {repo_id}...");
        let (owner, name) = repo_id.split_once('/').unwrap_or(("", repo_id));
        let client = HFClientSync::new()?;
        let weights_path = client.model(owner, name).download_file().filename("model.safetensors").send()?;
        eprintln!(
            "ESMFold2Runner: weights cached at {}",
            weights_path.display()
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], ESMFOLD2_DTYPE, &device)?
        };

        // TODO: also load biohub/ESMC-6B backbone (~12 GB) via ESMCRunner
        // let esmc = ESMCRunner::from_pretrained(ESMCModels::ESMC6B, device.clone())?;

        eprintln!("ESMFold2Runner: building model...");
        let model = ESMFold2Model::load(vb, config.clone())?;
        eprintln!("ESMFold2Runner: ready.");

        Ok(Self {
            model,
            config,
            device,
        })
    }

    /// Fold a single protein sequence.
    ///
    /// # Arguments
    /// * `sequence`           — amino-acid string (standard one-letter codes)
    /// * `num_loops`          — recycling iterations (typically 3)
    /// * `num_sampling_steps` — diffusion denoising steps (14 fast / 50 quality)
    ///
    /// # Note
    /// The ESMC-6B backbone (~12 GB) is not yet loaded.  Hidden states are
    /// initialised to zeros and will produce physically meaningless coordinates
    /// until the backbone is wired in.  The wiring is tracked separately.
    pub fn fold_protein(
        &self,
        sequence: &str,
        num_loops: usize,
        num_sampling_steps: usize,
    ) -> Result<ESMFold2Output> {
        let l = sequence.len();
        let device = &self.device;

        // Stub: backbone hidden states, zeros until ESMC-6B is wired in.
        // Shape [1, L, lm_d_model=2560].
        let hidden_states =
            Tensor::zeros(&[1, l, self.config.lm_d_model], ESMFOLD2_DTYPE, device)?;

        // Residue indices [1, L]: 0, 1, …, L-1.
        let res_idx: Vec<f32> = (0..l).map(|i| i as f32).collect();
        let residue_indices = Tensor::from_vec(res_idx, &[1, l], device)?;

        // Chain IDs [1, L]: all zeros (single chain).
        let chain_ids = Tensor::zeros(&[1, l], ESMFOLD2_DTYPE, device)?;

        Ok(self
            .model
            .forward(&hidden_states, &residue_indices, &chain_ids, num_loops, num_sampling_steps)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test — downloads biohub/ESMFold2-Fast (~755 MB) and runs a
    /// forward pass on ubiquitin (76 aa).  Skipped in CI; run with:
    ///   cargo test -p ferritin-plms test_esmfold2_fold_protein_integration -- --ignored
    #[test]
    #[ignore]
    fn test_esmfold2_fold_protein_integration() {
        let device = Device::Cpu;
        let runner =
            ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, device).expect("load failed");

        // Ubiquitin (76 aa)
        let seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
        let output = runner.fold_protein(seq, 1, 2).expect("fold_protein failed");

        let (_, l, d) = output.sample_atom_coords.dims3().unwrap();
        assert_eq!(l, seq.len(), "L must equal sequence length");
        assert_eq!(d, 3, "last dim must be 3 (x, y, z)");

        let plddt_dims = output.plddt.dims().to_vec();
        assert_eq!(plddt_dims, vec![1, seq.len()], "pLDDT shape mismatch");
    }
}
