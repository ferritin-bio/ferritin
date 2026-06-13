//! ferritin-plms
//!
//!
//! ```shell
//! cargo run --example amplify
//! cargo run --example amplify --features metal
//! ```
pub use amplify::amplify::{AMPLIFY, AmplifyOutput};
pub use amplify::amplify_runner::{AmplifyModels, AmplifyRunner};
pub use amplify::config::AMPLIFYConfig;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};
pub use esm2::esm2::{ESM2, ESM2Config};
pub use esm2::esm2_runner::{ESM2Models, ESM2Runner};
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

pub mod utils;
pub mod amplify;
pub mod esm2;
pub mod esmc;
pub mod esmfold2;
pub mod featurize;
pub mod ligandmpnn;
pub mod plm_runner;
pub mod types;
pub use plm_runner::PlmRunner;

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
