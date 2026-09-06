//! ESMFold2 pretrained model loading.
//!
//! Downloads the structure-head weights from HuggingFace and optionally wires
//! in the ESMC-6B backbone for end-to-end structure prediction.
//!
//! ## Pretrained loading is currently refused
//!
//! The ported module graph in this crate does **not** match the released
//! `biohub/ESMFold2-Fast` checkpoint: none of the checkpoint's 1032 tensors
//! resolve to a parameter of [`ESMFold2Model`]. This is not a prefix or
//! rename mismatch — the checkpoint's `lm_encoder` is a pair-track stack of
//! triangle-multiplication blocks, not the transformer adapter modelled here;
//! `folding_trunk` has no triangle attention; `structure_head` nests under
//! `diffusion_module` with AdaLN blocks and an atom decoder; and whole
//! components (`parcae_*`, `language_model.*`, `inputs_embedder.*`,
//! `rel_pos`, `z_init_1`/`z_init_2`, `token_bonds`) are not modelled at all.
//!
//! Both [`ESMFold2Runner::from_pretrained`] and
//! [`ESMFold2Runner::from_pretrained_with_backbone`] therefore return an error
//! naming that cause, rather than loading weights that would produce
//! meaningless coordinates. See `docs/decisions/esmfold2-port-mismatch.md`
//! for the full evidence and `docs/decisions/esmfold2-checkpoint-tensors.md`
//! for the checkpoint's real tensor inventory (ferritin-100.16).
//!
//! The layer implementations and [`ESMFold2Model::load`] itself remain usable
//! with a `VarBuilder` you supply (e.g. `VarBuilder::zeros`) for shape and
//! pipeline testing; only the pretrained path refuses.

use super::config::ESMFold2Config;
use super::model::ESMFold2Model;
use super::output::ESMFold2Output;
use crate::esmc::pretrained::ESMCRunner;
use crate::plm_runner::PlmRunner;
use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};

const ESMFOLD2_DTYPE: DType = DType::F32;

/// Why [`ESMFold2Runner::from_pretrained`] and
/// [`ESMFold2Runner::from_pretrained_with_backbone`] refuse to load.
///
/// Not one of the 1032 tensors in `biohub/ESMFold2-Fast` `model.safetensors`
/// resolves to a parameter of the ported [`ESMFold2Model`]: the checkpoint is
/// a different network from the one implemented here, so this is not a prefix
/// or rename mismatch that path-patching could fix.
pub const ARCHITECTURE_MISMATCH: &str = "\
ESMFold2 pretrained weights cannot be loaded: the ported architecture does not match the \
released checkpoint. None of the 1032 tensors in biohub/ESMFold2-Fast model.safetensors \
resolve to a parameter of this model — the checkpoint's lm_encoder is a pair-track stack of \
triangle-multiplication blocks (not a transformer adapter), folding_trunk has no triangle \
attention, structure_head nests under diffusion_module with AdaLN blocks and an atom decoder, \
and the parcae_*, language_model.*, inputs_embedder.*, rel_pos, z_init_1/z_init_2 and \
token_bonds components are not modelled at all. Loading is refused rather than patched, \
because resolving the paths without re-porting the forward pass would emit plausible-looking \
but meaningless coordinates. See docs/decisions/esmfold2-port-mismatch.md (ferritin-100.16). \
ESMFold2Model::load still works with a VarBuilder you supply, for shape testing.";

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
    /// Loading the pretrained structure head is **not supported** — this
    /// always returns an error (ferritin-100.16).
    ///
    /// The ported module graph does not match the released
    /// `biohub/ESMFold2-Fast` checkpoint, so there are no weights to load.
    /// See [`ARCHITECTURE_MISMATCH`] and the module docs for details.
    ///
    /// To exercise the pipeline with correct shapes, build an
    /// [`ESMFold2Model`] directly from a `VarBuilder` you control
    /// (e.g. `VarBuilder::zeros`) and construct the runner around it.
    pub fn from_pretrained(model_variant: ESMFold2Models, _device: Device) -> Result<Self> {
        // Validate the variant first so `Full` still reports its own reason.
        let (repo_id, _config) = model_variant.model_info()?;
        bail!("{ARCHITECTURE_MISMATCH} (requested repo: {repo_id})");
    }

    /// Loading the pretrained structure head plus the ESMC-6B backbone is
    /// **not supported** — this always returns an error (ferritin-100.16).
    ///
    /// The backbone itself loads fine ([`ESMCRunner`]); it is the ESMFold2
    /// structure head that has no loadable weights. See
    /// [`ARCHITECTURE_MISMATCH`].
    pub fn from_pretrained_with_backbone(
        model_variant: ESMFold2Models,
        _device: Device,
    ) -> Result<Self> {
        let (repo_id, _config) = model_variant.model_info()?;
        bail!("{ARCHITECTURE_MISMATCH} (requested repo: {repo_id})");
    }

    /// Build a runner around an already-constructed model.
    ///
    /// This is the only working way to obtain an [`ESMFold2Runner`] while
    /// pretrained loading is refused: supply your own `VarBuilder` to
    /// [`ESMFold2Model::load`]. With `backbone: None`, `fold_protein` feeds
    /// zeroed hidden states, so shapes are correct but coordinates are not
    /// physically meaningful.
    pub fn new(
        model: ESMFold2Model,
        config: ESMFold2Config,
        device: Device,
        backbone: Option<ESMCRunner>,
    ) -> Self {
        Self {
            model,
            config,
            device,
            backbone,
        }
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
            // `embed_residues` strips the backbone's special tokens per its own
            // declared layout, so this no longer hardcodes ESMC's BOS/EOS
            // (ferritin-100.7).
            Some(esmc) => esmc.embed_residues(sequence)?,
            None => Tensor::zeros(&[1, l, self.config.lm_d_model], ESMFOLD2_DTYPE, device)?,
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
            .map(|_| ())
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
        use candle_core::DType;
        use candle_nn::VarBuilder;

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
        let runner = ESMFold2Runner::new(model, cfg, device.clone(), None);

        let seq = "ACDEFGHIK"; // 9 residues
        let out = runner.fold_protein(seq, 1, 2).unwrap();

        assert_eq!(out.sample_atom_coords.dims(), &[1, seq.len(), 3]);
        assert_eq!(out.plddt.dims(), &[1, seq.len()]);
        assert_eq!(out.ptm.dims(), &[1]);
        assert_eq!(out.iptm.dims(), &[1]);
    }

    /// Pretrained loading is refused with an explanation, not with a bare
    /// `cannot find tensor ...` (ferritin-100.16).
    ///
    /// This replaces the former `test_esmfold2_fold_stub_integration` and
    /// `test_esmfold2_fold_with_backbone` integration tests, which downloaded
    /// ~755 MB (and ~12 GB) only to fail on the first missing tensor. Neither
    /// can pass until the port is rewritten against the real checkpoint, so
    /// they assert the refusal instead — and need no network to do it.
    #[test]
    fn test_from_pretrained_refuses_with_reason() {
        for err in [
            ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, Device::Cpu)
                .map(|_| ())
                .expect_err("Fast must refuse: the port does not match the checkpoint"),
            ESMFold2Runner::from_pretrained_with_backbone(ESMFold2Models::Fast, Device::Cpu)
                .map(|_| ())
                .expect_err("Fast+backbone must refuse: the port does not match the checkpoint"),
        ] {
            let msg = err.to_string();
            assert!(
                msg.contains("does not match the released checkpoint"),
                "error should name the architecture mismatch; got: {msg}"
            );
            assert!(
                msg.contains("biohub/ESMFold2-Fast"),
                "error should name the checkpoint; got: {msg}"
            );
        }
    }

    /// `Full` keeps reporting its own unsupported-variant reason (ferritin-100.4)
    /// rather than being masked by the architecture-mismatch refusal.
    #[test]
    fn test_from_pretrained_full_reports_variant_error() {
        let err = ESMFold2Runner::from_pretrained(ESMFold2Models::Full, Device::Cpu)
            .map(|_| ())
            .expect_err("Full must refuse");
        assert!(
            err.to_string().contains("not yet supported"),
            "Full should report the unsupported-variant error; got: {err}"
        );
    }
}
