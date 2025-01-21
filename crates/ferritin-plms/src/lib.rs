//! ferritin-plms
//!
//!
//! ```shell
//! cargo run --example amplify
//! cargo run --example amplify --features metal
//! ```
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result, Tensor};
pub use amplify::amplify::AMPLIFY;
pub use amplify::amplify_runner::{AmplifyModels, AmplifyRunner};
pub use amplify::config::AMPLIFYConfig;
pub use amplify::outputs::ModelOutput;
pub use esm::models::esmc::{ESMCConfig, ESMC};
pub use esm2::models::esm2::{ESM2Config, ESM2};
pub use ligandmpnn::configs::ProteinMPNNConfig;
pub use ligandmpnn::model::ProteinMPNN;
pub use ligandmpnn::proteinfeatures::LMPNNFeatures;

pub mod amplify;
pub mod esm;
pub mod esm2;
pub mod ligandmpnn;
pub mod types;




pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
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
