//! ferritin-amplify
//!
//!
//! - [amplify model](https://github.com/chandar-lab/AMPLIFY)
//! - [amplify hf - 120M](https://huggingface.co/chandar-lab/AMPLIFY_120M)
//!
//! To try and example of this model:
//!
//! ```shell
//! cargo run --example amplify
//! cargo run --example amplify --features metal
//! ```
//!
//!
//!
pub use amplify::AMPLIFY;
pub use amplify::config::AMPLIFYConfig;
pub use amplify::outputs::ModelOutput;
pub use esm::models::esmc::{ESMCConfig, ESMC};
pub use esm2::models::esm2::{ESM2Config, ESM2};

pub mod amplify;
pub mod esm;
pub mod esm2;

