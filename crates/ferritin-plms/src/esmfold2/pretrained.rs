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
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::Api};

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
        let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, "main".to_string());
        let api = Api::new()?;
        let weights_path = api.repo(repo).get("model.safetensors")?;
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
    /// * `sequence` — amino-acid string (standard one-letter codes)
    /// * `num_loops` — recycling iterations (default 3)
    /// * `num_sampling_steps` — diffusion denoising steps (default 50 for quality, 14 for speed)
    ///
    /// # Note
    /// This method is a stub. The forward pass is not yet implemented.
    /// Each layer in `src/esmfold2/layers/` has a working scaffold with the
    /// correct type signatures and VarBuilder paths; the mathematical
    /// implementations are the remaining work.
    pub fn fold_protein(
        &self,
        _sequence: &str,
        _num_loops: usize,
        _num_sampling_steps: usize,
    ) -> Result<ESMFold2Output> {
        anyhow::bail!(
            "ESMFold2 forward pass is not yet implemented.\n\
             Remaining work per layer:\n\
             - LMEncoder: 4-layer transformer blocks + 2560→384 projection\n\
             - FoldingTrunk: 24-layer Pairformer (row/column attention + triangle updates)\n\
             - DiffusionModule: AF3-style EDM loop (token 12-block + atom 3-block)\n\
             - ConfidenceHead: pLDDT/pAE bins → scalar confidence\n\
             See src/esmfold2/layers/ for the scaffolded stubs."
        )
    }
}
