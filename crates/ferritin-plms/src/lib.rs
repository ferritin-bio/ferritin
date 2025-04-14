//! ferritin-plms
//!
//!
//! ```shell
//! cargo run --example amplify
//! cargo run --example amplify --features metal
//! ```
pub use amplify::amplify::{AMPLIFY, ModelOutput};
pub use amplify::amplify_runner::{AmplifyModels, AmplifyRunner};
pub use amplify::config::AMPLIFYConfig;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};

pub use esm::models::esmc::{ESMC, ESMCConfig};
pub use esm2::esm2::{ESM2, ESM2Config};
pub use ligandmpnn::configs::ProteinMPNNConfig;
pub use ligandmpnn::model::ProteinMPNN;
pub use ligandmpnn::proteinfeatures::LMPNNFeatures;

pub mod amplify;
pub mod esm;
pub mod esm2;
pub mod ligandmpnn;
pub mod types;

/// Returns the best available device for computation.
///
/// Prioritizes CUDA GPU if available, then Metal GPU on supported platforms,
/// and falls back to CPU if no GPU acceleration is available.
pub fn device() -> Result<Device> {
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
