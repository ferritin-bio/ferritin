//! ferritin-plms
//!
//!
//! ```shell
//! cargo run --example amplify
//! cargo run --example amplify --features metal
//! ```
//!
//! # Model support matrix
//!
//! Generated from [`registry::REGISTRY`]; regenerate with
//! `cargo test -p ferritin-plms --lib print_support_matrix -- --ignored --nocapture`.
//! `test_lib_rs_support_matrix_is_current` fails if this copy drifts.
//!
//! The column to read first is **Parity**. "It compiles" and even "it loads"
//! are not what you need before trusting a number — you need to know whether
//! anyone has compared this port's output against the reference
//! implementation. Two models have. The rest are unverified: not known to be
//! wrong, but nothing proves them right.
//!
//! <!-- BEGIN SUPPORT MATRIX -->
//! | Model | Family | Weights | Parity | Status |
//! |---|---|---|---|---|
//! | `esm2-t6-8m` | Esm2 | `facebook/esm2_t6_8M_UR50D` (safetensors) | verified (`esm2_parity`) | supported |
//! | `esm2-t12-35m` | Esm2 | `facebook/esm2_t12_35M_UR50D` (safetensors) | **not checked** | supported |
//! | `esm2-t30-150m` | Esm2 | `facebook/esm2_t30_150M_UR50D` (safetensors) | **not checked** | supported |
//! | `esm2-t33-650m` | Esm2 | `facebook/esm2_t33_650M_UR50D` (safetensors) | **not checked** | supported |
//! | `esm2-t36-3b` | Esm2 | `facebook/esm2_t36_3B_UR50D` (safetensors) | **not checked** | supported |
//! | `esm2-t48-15b` | Esm2 | `facebook/esm2_t48_15B_UR50D` (safetensors) | **not checked** | supported |
//! | `esm1v-t33-650m-ur90s-1` | Esm2 | `facebook/esm1v_t33_650M_UR90S_1` (pth) | **not checked** | supported |
//! | `esm1v-t33-650m-ur90s-2` | Esm2 | `facebook/esm1v_t33_650M_UR90S_2` (pth) | **not checked** | supported |
//! | `esm1v-t33-650m-ur90s-3` | Esm2 | `facebook/esm1v_t33_650M_UR90S_3` (pth) | **not checked** | supported |
//! | `esm1v-t33-650m-ur90s-4` | Esm2 | `facebook/esm1v_t33_650M_UR90S_4` (pth) | **not checked** | supported |
//! | `esm1v-t33-650m-ur90s-5` | Esm2 | `facebook/esm1v_t33_650M_UR90S_5` (pth) | **not checked** | supported |
//! | `esm1b-t33-650m-ur50s` | Esm2 | `facebook/esm1b_t33_650M_UR50S` (pth) | **not checked** | supported |
//! | `saprot-35m-af2` | Esm2 | `westlake-repl/SaProt_35M_AF2` (pth) | **not checked** | supported |
//! | `saprot-650m-af2` | Esm2 | `westlake-repl/SaProt_650M_AF2` (pth) | **not checked** | supported |
//! | `fastesm2-650` | Esm2 | `Synthyra/FastESM2_650` (safetensors) | **not checked** | supported |
//! | `pepmlm-650m` | Esm2 | `ChatterjeeLab/PepMLM-650M` (pth) | **not checked** | supported |
//! | `dplm-650m` | Esm2 | `airkingbd/dplm_650m` (pth) | **not checked** | supported |
//! | `amplify-120m` | Amplify | `chandar-lab/AMPLIFY_120M` (safetensors) | verified (`amplify_parity`) | supported |
//! | `amplify-350m` | Amplify | `chandar-lab/AMPLIFY_350M` (safetensors) | **not checked** | supported |
//! | `esmc-300m` | Esmc | `EvolutionaryScale/esmc-300m-2024-12` (pth) | **not checked** | supported |
//! | `esmc-600m` | Esmc | `EvolutionaryScale/esmc-600m-2024-12` (pth) | **not checked** | supported |
//! | `esmc-6b` | Esmc | `EvolutionaryScale/esmc-6b-2024-12` (safetensors) | **not checked** | supported |
//! | `esm3-sm-open-v1` | Esm3 | `EvolutionaryScale/esm3-sm-open-v1` (pth) | **not checked** | supported |
//! | `esm3-structure-encoder-v0` | Esm3 | `EvolutionaryScale/esm3-sm-open-v1` (pth) | **not checked** | **unsupported** — the ported VQ-VAE encoder is a different shape from the released checkpoint (ferritin-100.22) |
//! | `esmfold2-fast` | Esmfold2 | `biohub/ESMFold2-Fast` (safetensors) | **not checked** | **unsupported** — the ported architecture does not match the released checkpoint (ferritin-100.17) |
//! | `proteinmpnn-v48-020` | Mpnn | `zcpbx/ligandmpnn-weights` (pth) | **not checked** | supported |
//! <!-- END SUPPORT MATRIX -->

// The crate deliberately uses the `foo/mod.rs` + inner `mod foo` layout for
// each model family, so module_inception is expected throughout.
#![allow(clippy::module_inception)]

pub use amplify::amplify::{AMPLIFY, AmplifyOutput};
pub use amplify::amplify_runner::{AmplifyModels, AmplifyRunner};
pub use amplify::config::AMPLIFYConfig;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};
pub use esm2::esm2::{ESM2, ESM2Config};
pub use esm2::esm2_runner::{ESM2Models, ESM2Runner};
pub use esm3::models::esm3::ESM3Config;
pub use esm3::pretrained::{ESM3Models, ESM3Runner};
pub use esmc::models::esmc::{ESMC, ESMCConfig, ESMCOutput, LogitsConfig, LogitsOutput};
pub use esmc::pretrained::{ESMCModels, ESMCRunner};
pub use esmfold2::config::ESMFold2Config;
pub use esmfold2::output::ESMFold2Output;
pub use esmfold2::pretrained::{ESMFold2Models, ESMFold2Runner};
pub use esmfold2::{
    ChainInput, DNAInput, LigandInput, Modification, ProteinInput, StructurePredictionInput,
};
pub use featurize::StructureFeatures;
pub use ligandmpnn::configs::ProteinMPNNConfig;
pub use ligandmpnn::model::ProteinMPNN;
pub use ligandmpnn::pmpnn_runner::{ProteinMPNNModels, ProteinMPNNRunner};

pub mod amplify;
pub mod esm2;
pub mod esm3;
pub mod esmc;
pub mod esmfold2;
pub mod featurize;
pub mod ligandmpnn;
pub mod loader;
pub mod plm_runner;
pub mod registry;
pub mod types;
pub mod utils;
pub use plm_runner::{ModelMetadata, PlmRunner, SpecialTokenLayout};
pub use registry::{Family, ModelCard, ParityStatus, REGISTRY, TokenizerSpec};

/// Returns the best available device for computation.
///
/// If `cpu` is true, always returns `Device::Cpu` regardless of available hardware.
/// Otherwise prioritizes CUDA GPU if available, then Metal GPU on supported platforms,
/// and falls back to CPU if no GPU acceleration is available.
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
