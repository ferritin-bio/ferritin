//! ESMFold2 pretrained model loading.
//!
//! Downloads the structure-head weights from HuggingFace and optionally wires
//! in the ESMC-6B backbone for end-to-end structure prediction.
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
//! ## Two loading modes
//!
//! - [`ESMFold2Runner::from_pretrained`] — structure head only (~755 MB).
//!   Backbone hidden states are zero-initialised; coordinates will be
//!   physically meaningless but all shapes are correct (useful for shape testing).
//!
//! - [`ESMFold2Runner::from_pretrained_with_backbone`] — structure head +
//!   ESMC-6B backbone (~12 GB total). Required for real structure predictions.

use super::config::ESMFold2Config;
use super::model::ESMFold2Model;
use super::output::ESMFold2Output;
use crate::esmc::pretrained::{ESMCModels, ESMCRunner};
use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::HFClientSync;

const ESMFOLD2_DTYPE: DType = DType::F32;

// ── ESMFold2Models enum ───────────────────────────────────────────────────────

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
    ///
    /// `Full` currently returns an error: only the Fast architecture/config is
    /// ported. The Full variant adds MSA conditioning (`msa_encoder.*` weights
    /// that Fast disables), so pairing the Full repo with `ESMFold2Config::fast()`
    /// would silently load a subset of the weights and emit degraded structures.
    /// Refusing is the honest failure until a `full()` config is verified against
    /// the reference implementation (ferritin-100.4).
    pub fn model_info(&self) -> Result<(&'static str, ESMFold2Config)> {
        match self {
            Self::Fast => Ok(("biohub/ESMFold2-Fast", ESMFold2Config::fast())),
            Self::Full => bail!(
                "ESMFold2Models::Full is not yet supported: only the Fast config is ported. \
                 The Full variant (MSA conditioning) needs a config verified against the \
                 reference before it can load without silently degrading structures."
            ),
        }
    }
}

// ── ESMFold2Runner ────────────────────────────────────────────────────────────

/// Wraps a loaded ESMFold2 model for all-atom structure prediction inference.
///
/// Create with:
/// - [`from_pretrained`][Self::from_pretrained] — structure head only (~755 MB),
///   backbone stubs with zeros (shape testing only).
/// - [`from_pretrained_with_backbone`][Self::from_pretrained_with_backbone] —
///   structure head + ESMC-6B backbone (~12 GB total, real predictions).
pub struct ESMFold2Runner {
    model: ESMFold2Model,
    config: ESMFold2Config,
    device: Device,
    /// ESMC-6B backbone. `None` → hidden states are zeroed (stub mode).
    backbone: Option<ESMCRunner>,
}

impl ESMFold2Runner {
    /// Download and load the ESMFold2 structure head only (~755 MB).
    ///
    /// The ESMC-6B backbone is **not** loaded; hidden states passed to
    /// `fold_protein` will be all-zeros, so predicted coordinates are
    /// physically meaningless. Use this for shape/pipeline testing without
    /// requiring 12 GB of disk space.
    pub fn from_pretrained(model_variant: ESMFold2Models, device: Device) -> Result<Self> {
        let (repo_id, config) = model_variant.model_info()?;
        let model = Self::load_structure_head(repo_id, &config, &device)?;
        Ok(Self { model, config, device, backbone: None })
    }

    /// Download and load the structure head (~755 MB) **and** the ESMC-6B
    /// backbone (~12 GB).  Required for predictions with meaningful pLDDT.
    pub fn from_pretrained_with_backbone(
        model_variant: ESMFold2Models,
        device: Device,
    ) -> Result<Self> {
        let (repo_id, config) = model_variant.model_info()?;
        let model = Self::load_structure_head(repo_id, &config, &device)?;
        eprintln!("ESMFold2Runner: loading ESMC-6B backbone (~12 GB)...");
        let backbone = ESMCRunner::from_pretrained(ESMCModels::ESMC6B, device.clone())?;
        eprintln!("ESMFold2Runner: backbone ready.");
        Ok(Self { model, config, device, backbone: Some(backbone) })
    }

    fn load_structure_head(
        repo_id: &str,
        config: &ESMFold2Config,
        device: &Device,
    ) -> Result<ESMFold2Model> {
        eprintln!("ESMFold2Runner: downloading structure head from {repo_id}...");
        let (owner, name) = repo_id.split_once('/').unwrap_or(("", repo_id));
        let client = HFClientSync::new()?;
        let weights_path = client
            .model(owner, name)
            .download_file()
            .filename("model.safetensors")
            .send()?;
        eprintln!("ESMFold2Runner: weights at {}", weights_path.display());

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], ESMFOLD2_DTYPE, device)?
        };
        Ok(ESMFold2Model::load(vb, config.clone())?)
    }

    /// Fold a single protein sequence.
    ///
    /// # Arguments
    /// * `sequence`           — amino-acid string (standard one-letter codes)
    /// * `num_loops`          — recycling iterations (typically 3)
    /// * `num_sampling_steps` — diffusion denoising steps (14 fast / 50 quality)
    ///
    /// When the runner was created with [`from_pretrained`][Self::from_pretrained]
    /// (no backbone), hidden states are zeros and coordinates will be arbitrary.
    /// Use [`from_pretrained_with_backbone`][Self::from_pretrained_with_backbone]
    /// for real predictions.
    pub fn fold_protein(
        &self,
        sequence: &str,
        num_loops: usize,
        num_sampling_steps: usize,
    ) -> Result<ESMFold2Output> {
        let l = sequence.len();
        let device = &self.device;

        // ESMC-6B backbone → (1, L, 2560).
        // When backbone is None, use zeros (stub for shape testing).
        let hidden_states = match &self.backbone {
            Some(esmc) => {
                // embed_sequence returns (1, L+2, d_model) with BOS and EOS tokens.
                let embs = esmc.embed_sequence(sequence)?;
                // Strip BOS (index 0) and EOS (index L+1), keeping L residues.
                embs.narrow(1, 1, l)?
            }
            None => Tensor::zeros(
                &[1, l, self.config.lm_d_model],
                ESMFOLD2_DTYPE,
                device,
            )?,
        };

        let res_idx: Vec<f32> = (0..l).map(|i| i as f32).collect();
        let residue_indices = Tensor::from_vec(res_idx, &[1, l], device)?;
        let chain_ids = Tensor::zeros(&[1, l], ESMFOLD2_DTYPE, device)?;

        Ok(self.model.forward(
            &hidden_states,
            &residue_indices,
            &chain_ids,
            num_loops,
            num_sampling_steps,
        )?)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `model_info` yields the Fast config for Fast and refuses Full rather than
    /// silently pairing the Full repo with the Fast config (ferritin-100.4).
    #[test]
    fn test_model_info_fast_ok_full_errors() {
        let (repo, _cfg) = ESMFold2Models::Fast
            .model_info()
            .expect("Fast should return a config");
        assert_eq!(repo, "biohub/ESMFold2-Fast");

        let err = ESMFold2Models::Full
            .model_info()
            .expect_err("Full must not load with the Fast config");
        assert!(
            err.to_string().contains("not yet supported"),
            "Full error should explain it is unsupported; got: {err}"
        );
    }

    /// Shape smoke-test with stub backbone (no download needed).
    /// Uses a tiny config so the test runs in milliseconds.
    #[test]
    fn test_fold_protein_stub_shapes() {
        use super::super::config::ESMFold2Config;
        use candle_nn::VarBuilder;
        use candle_core::DType;

        let device = Device::Cpu;
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
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ESMFold2Model::load(vb, cfg.clone()).unwrap();
        let runner = ESMFold2Runner { model, config: cfg, device: device.clone(), backbone: None };

        let seq = "ACDEFGHIK"; // 9 residues
        let out = runner.fold_protein(seq, 1, 2).unwrap();

        assert_eq!(out.sample_atom_coords.dims(), &[1, seq.len(), 3]);
        assert_eq!(out.plddt.dims(), &[1, seq.len()]);
        assert_eq!(out.ptm.dims(), &[1]);
        assert_eq!(out.iptm.dims(), &[1]);
    }

    /// Integration test: downloads structure head (~755 MB) and folds ubiquitin.
    /// The backbone is stubbed with zeros so coordinates are not meaningful,
    /// but all shapes and the full pipeline must succeed.
    ///
    /// Run with:
    ///   cargo test -p ferritin-plms test_esmfold2_fold_stub_integration -- --ignored
    #[test]
    #[ignore = "requires downloading biohub/ESMFold2-Fast weights (~755 MB)"]
    fn test_esmfold2_fold_stub_integration() {
        let device = Device::Cpu;
        let runner =
            ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, device).expect("load failed");

        // Ubiquitin (76 aa)
        let seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
        let out = runner.fold_protein(seq, 1, 2).expect("fold_protein failed");

        assert_eq!(out.sample_atom_coords.dims(), &[1, seq.len(), 3]);
        assert_eq!(out.plddt.dims(), &[1, seq.len()]);
    }

    /// Full end-to-end integration: ESMC-6B backbone + structure head.
    /// Folds ubiquitin and asserts pLDDT > 0.5 on average (backbone gives
    /// real signal; exact quality depends on diffusion steps).
    ///
    /// Run with:
    ///   cargo test -p ferritin-plms test_esmfold2_fold_with_backbone -- --ignored
    #[test]
    #[ignore = "requires biohub/ESMFold2-Fast (~755 MB) + biohub/ESMC-6B (~12 GB)"]
    fn test_esmfold2_fold_with_backbone() -> Result<()> {
        let device = Device::Cpu;
        let runner =
            ESMFold2Runner::from_pretrained_with_backbone(ESMFold2Models::Fast, device)?;

        let seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
        let out = runner.fold_protein(seq, 3, 14)?;

        assert_eq!(out.sample_atom_coords.dims(), &[1, seq.len(), 3]);
        assert_eq!(out.plddt.dims(), &[1, seq.len()]);

        let mean_plddt = out.plddt.mean_all()?.to_scalar::<f32>()?;
        assert!(mean_plddt > 0.5, "expected mean pLDDT > 0.5, got {mean_plddt:.3}");

        Ok(())
    }
}
