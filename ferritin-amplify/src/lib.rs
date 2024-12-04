//! ferritin-amplify
//! - [amplify model](https://github.com/chandar-lab/AMPLIFY)
//! - [amplify hf - 120M](https://huggingface.co/chandar-lab/AMPLIFY_120M)
//!

mod amplify;
pub use amplify::amplify::{AMPLIFYConfig, ModelOutput, AMPLIFY};
pub use amplify::tokenizer::ProteinTokenizer;
