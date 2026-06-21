//! ESM3 multi-track tokenization.
//!
//! | Track     | Vocab  | Layout                                      |
//! |-----------|--------|---------------------------------------------|
//! | sequence  | 33     | `<cls>=0 <pad>=1 <eos>=2 ... <mask>=32`    |
//! | structure | 4101   | VQ-VAE 0..4095 + MASK/EOS/BOS/PAD/CB       |
//! | ss8       | 11     | PAD=0 MASK=1 UNK=2 GHITEBSC=3..10         |
//! | sasa      | 19     | PAD=0 MASK=1 UNK=2 bins=3..18             |
//!
//! Function/InterPro tokenization requires external data files (future work).

pub mod sasa;
pub mod sequence;
pub mod ss8;
pub mod structure;

pub use sasa::tokenize_sasa;
pub use sequence::{decode_sequence, tokenize_sequence};
pub use ss8::tokenize_ss8;
pub use structure::tokenize_structure;
