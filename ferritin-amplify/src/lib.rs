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
mod amplify;
pub use amplify::amplify::AMPLIFY;
pub use amplify::config::AMPLIFYConfig;
pub use amplify::outputs::ModelOutput;
