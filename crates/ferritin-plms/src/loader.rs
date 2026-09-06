//! One place to describe where a model's weights live and how to load them.
//!
//! Before this module every runner reimplemented the same download block:
//! split the repo id, build an [`HFClientSync`], download a file, then either
//! `unsafe { VarBuilder::from_mmaped_safetensors(..) }` or
//! `PthTensors::new(..)`. Six near-identical copies drifted apart — only ESM3
//! attached any error context, and every one of them inherited the same
//! silently-wrong repo-id split (`split_once('/').unwrap_or(("", repo_id))`,
//! which turns a malformed id into an empty owner and a confusing 404 rather
//! than a clear error).
//!
//! The pieces here are:
//!
//! - [`WeightSource`] — plain `const` data on each model enum: which repo,
//!   which revision, and which on-disk [`Format`] the weights use.
//! - [`LoadOptions`] — the device and dtype to load onto, so the six
//!   per-module `const *_DTYPE` definitions collapse into one field.
//! - [`WeightSource::var_builder`] — the single place holding the `unsafe`
//!   mmap.
//! - [`optional_prefix`] — the "is this checkpoint wrapped in an HF
//!   `*ForMaskedLM` class?" probe, generalised from ESMC's `esmc.` special
//!   case.
//!
//! ```no_run
//! # use ferritin_plms::loader::{LoadOptions, WeightSource};
//! # use candle_core::Device;
//! const WEIGHTS: WeightSource = WeightSource::safetensors("facebook/esm2_t6_8M_UR50D");
//!
//! let opts = LoadOptions::new(Device::Cpu);
//! let vb = WEIGHTS.var_builder("model.safetensors", &opts)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::{Context, Result, bail};
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{HFClientSync, HFRepositorySync, RepoTypeModel};
use std::path::{Path, PathBuf};

// ── LoadOptions ───────────────────────────────────────────────────────────────

/// Device and dtype to load a model onto.
///
/// Defaults to `F32`, which is what every runner used before this existed.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Element type to materialise weights as.
    pub dtype: DType,
    /// Device to place weights on.
    pub device: Device,
}

impl LoadOptions {
    /// `F32` on `device`.
    pub fn new(device: Device) -> Self {
        Self {
            dtype: DType::F32,
            device,
        }
    }

    /// Override the dtype (builder style).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
}

// ── Format ────────────────────────────────────────────────────────────────────

/// How a checkpoint is stored on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// `.safetensors`, loaded by mmap.
    Safetensors,
    /// PyTorch `.pth`/`.pt` pickle.
    Pth {
        /// Sub-dictionary holding the tensors, when the checkpoint nests them
        /// (e.g. ProteinMPNN's `model_state_dict`). `None` = tensors at the root.
        root_key: Option<&'static str>,
    },
}

// ── WeightSource ──────────────────────────────────────────────────────────────

/// Where a model's weights live on the HuggingFace hub, and how to read them.
///
/// Intended to be a `const` on each model enum:
///
/// ```
/// # use ferritin_plms::loader::WeightSource;
/// const AMP120M: WeightSource = WeightSource::safetensors("chandar-lab/AMPLIFY_120M");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WeightSource {
    /// Full `owner/name` repo id.
    pub repo_id: &'static str,
    /// Git revision; `None` means the hub default (`main`).
    pub revision: Option<&'static str>,
    /// On-disk format of the weight files.
    pub format: Format,
}

impl WeightSource {
    /// A safetensors repo at the default revision.
    pub const fn safetensors(repo_id: &'static str) -> Self {
        Self {
            repo_id,
            revision: None,
            format: Format::Safetensors,
        }
    }

    /// A PyTorch-pickle repo at the default revision.
    ///
    /// `root_key` names the sub-dictionary holding the tensors, or `None` if
    /// they sit at the root.
    pub const fn pth(repo_id: &'static str, root_key: Option<&'static str>) -> Self {
        Self {
            repo_id,
            revision: None,
            format: Format::Pth { root_key },
        }
    }

    /// Pin to a specific revision.
    pub const fn at_revision(mut self, revision: &'static str) -> Self {
        self.revision = Some(revision);
        self
    }

    /// Split and validate `repo_id` into `(owner, name)`.
    ///
    /// Unlike `split_once('/').unwrap_or(("", repo_id))` — which every runner
    /// used to do — a malformed id is an error here rather than an empty owner
    /// and a 404 several seconds later.
    fn owner_and_name(&self) -> Result<(&'static str, &'static str)> {
        let (owner, name) = self.repo_id.split_once('/').ok_or_else(|| {
            anyhow::anyhow!(
                "malformed HuggingFace repo id {:?}: expected 'owner/name'",
                self.repo_id
            )
        })?;
        if owner.is_empty() || name.is_empty() || name.contains('/') {
            bail!(
                "malformed HuggingFace repo id {:?}: expected 'owner/name' with both parts non-empty",
                self.repo_id
            );
        }
        Ok((owner, name))
    }

    fn repo(&self) -> Result<HFRepositorySync<RepoTypeModel>> {
        let (owner, name) = self.owner_and_name()?;
        let client = HFClientSync::new().with_context(|| {
            format!(
                "failed to initialise the HuggingFace client while loading {}",
                self.repo_id
            )
        })?;
        Ok(client.model(owner, name))
    }

    /// Download `filename`, returning its cached path.
    ///
    /// Errors name both the repo and the file — previously only ESM3 did this,
    /// so a failure elsewhere surfaced as a bare transport error.
    pub fn fetch(&self, filename: &str) -> Result<PathBuf> {
        self.repo()?
            .download_file()
            .filename(filename.to_string())
            .maybe_revision(self.revision.map(str::to_string))
            .send()
            .with_context(|| format!("failed to download {filename} from {}", self.repo_id))
    }

    /// Download `filename`, returning `None` if it is absent or unreachable.
    ///
    /// For genuinely optional files such as ESM-2's `config.json`, where the
    /// runner falls back to a built-in config.
    pub fn fetch_optional(&self, filename: &str) -> Option<PathBuf> {
        self.fetch(filename).ok()
    }

    /// Download `filename` and build a [`VarBuilder`] over it.
    ///
    /// This is the only place in the crate that performs the safetensors mmap,
    /// and the only place the dtype is applied.
    pub fn var_builder(&self, filename: &str, opts: &LoadOptions) -> Result<VarBuilder<'static>> {
        let path = self.fetch(filename)?;
        self.var_builder_from_path(&path, opts)
    }

    /// Build a [`VarBuilder`] over an already-downloaded (or local) file.
    ///
    /// Used by loaders that also accept a path directly, such as
    /// `ProteinMPNNRunner::from_path`.
    pub fn var_builder_from_path(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<VarBuilder<'static>> {
        var_builder_from_path(path, self.format, opts)
    }
}

/// Build a [`VarBuilder`] over a local weight file of the given [`Format`].
///
/// Free-standing counterpart to [`WeightSource::var_builder_from_path`], for
/// paths with no associated hub repo.
pub fn var_builder_from_path(
    path: &Path,
    format: Format,
    opts: &LoadOptions,
) -> Result<VarBuilder<'static>> {
    match format {
        Format::Safetensors => {
            // SAFETY: mmap of a file we just materialised in the HF cache. As
            // everywhere in candle, this assumes nothing else mutates the file
            // while it is mapped.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[path], opts.dtype, &opts.device)
                    .with_context(|| format!("failed to mmap {}", path.display()))?
            };
            Ok(vb)
        }
        Format::Pth { root_key } => {
            let pth = PthTensors::new(path, root_key)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            Ok(VarBuilder::from_backend(
                Box::new(pth),
                opts.dtype,
                opts.device.clone(),
            ))
        }
    }
}

// ── Optional wrapper prefix ───────────────────────────────────────────────────

/// Descend into `prefix` when the checkpoint nests the backbone under it.
///
/// HuggingFace wrapper classes (`EsmForMaskedLM`, `ESMCForMaskedLM`, …) store
/// the backbone under an attribute, so the same architecture ships both flat
/// and prefixed. `probe` is a tensor that always exists in the backbone; if
/// `{prefix}.{probe}` is present the prefixed root is returned, otherwise `vb`
/// is returned unchanged.
///
/// ```no_run
/// # use ferritin_plms::loader::optional_prefix;
/// # fn f(vb: candle_nn::VarBuilder<'static>) {
/// // "esmc.embed.weight" present → root at "esmc", else flat.
/// let root = optional_prefix(vb, "esmc", "embed.weight");
/// # }
/// ```
pub fn optional_prefix<'a>(vb: VarBuilder<'a>, prefix: &str, probe: &str) -> VarBuilder<'a> {
    if vb.contains_tensor(&format!("{prefix}.{probe}")) {
        vb.pp(prefix)
    } else {
        vb
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owner_and_name_splits_valid_id() {
        let src = WeightSource::safetensors("chandar-lab/AMPLIFY_120M");
        assert_eq!(
            src.owner_and_name().unwrap(),
            ("chandar-lab", "AMPLIFY_120M")
        );
    }

    /// The old `split_once('/').unwrap_or(("", repo_id))` turned each of these
    /// into an empty owner and a confusing 404 after a network round-trip.
    #[test]
    fn test_owner_and_name_rejects_malformed_ids() {
        for bad in ["no-slash", "/leading", "trailing/", "a/b/c", ""] {
            let err = WeightSource::safetensors(bad)
                .owner_and_name()
                .expect_err("{bad} should be rejected");
            assert!(
                err.to_string().contains("malformed HuggingFace repo id"),
                "error should name the malformed id for {bad:?}; got: {err}"
            );
        }
    }

    #[test]
    fn test_revision_defaults_to_none_and_is_pinnable() {
        let src = WeightSource::safetensors("owner/name");
        assert_eq!(src.revision, None);
        assert_eq!(src.at_revision("v2").revision, Some("v2"));
    }

    #[test]
    fn test_format_constructors() {
        assert_eq!(WeightSource::safetensors("o/n").format, Format::Safetensors);
        assert_eq!(
            WeightSource::pth("o/n", Some("model_state_dict")).format,
            Format::Pth {
                root_key: Some("model_state_dict")
            }
        );
    }

    #[test]
    fn test_load_options_defaults_to_f32() {
        let opts = LoadOptions::new(Device::Cpu);
        assert_eq!(opts.dtype, DType::F32);
        assert_eq!(opts.with_dtype(DType::F16).dtype, DType::F16);
    }

    /// `optional_prefix` picks the prefixed root only when the probe resolves.
    #[test]
    fn test_optional_prefix_falls_back_when_absent() {
        use candle_core::Tensor;
        use std::collections::HashMap;

        let device = Device::Cpu;
        let mut flat = HashMap::new();
        flat.insert(
            "embed.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(flat, DType::F32, &device);
        // No "esmc." prefix present → unchanged, so "embed.weight" resolves.
        let root = optional_prefix(vb, "esmc", "embed.weight");
        assert!(root.contains_tensor("embed.weight"));
    }

    #[test]
    fn test_optional_prefix_descends_when_present() {
        use candle_core::Tensor;
        use std::collections::HashMap;

        let device = Device::Cpu;
        let mut wrapped = HashMap::new();
        wrapped.insert(
            "esmc.embed.weight".to_string(),
            Tensor::zeros((2, 2), DType::F32, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(wrapped, DType::F32, &device);
        let root = optional_prefix(vb, "esmc", "embed.weight");
        assert!(
            root.contains_tensor("embed.weight"),
            "after descending into 'esmc' the leaf should resolve unprefixed"
        );
    }
}
