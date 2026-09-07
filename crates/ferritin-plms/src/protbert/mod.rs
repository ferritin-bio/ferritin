//! ProtBert — BERT over an amino-acid alphabet (ferritin-goh.7).
//!
//! `Rostlab/prot_bert` and `Rostlab/prot_bert_bfd` are plain BERT:
//! `BertForMaskedLM`, hidden 1024, 30 layers, 16 heads, a 30-token vocabulary
//! of single amino-acid letters. `candle_transformers::models::bert` already
//! implements the architecture, so this module adds no attention or FFN code —
//! only a config mapping and the [`tokenizer`].
//!
//! # Status: the weights cannot currently be loaded
//!
//! Both repos publish `pytorch_model.bin` in PyTorch's **legacy**
//! (pre-1.6) pickle container, not the zip-based one. `candle`'s `PthTensors`
//! opens a `.pth` as a zip archive, so it rejects these files outright with
//! `invalid Zip archive: Could not find EOCD`. Neither repo ships safetensors,
//! and no upstream mirror does either.
//!
//! What that leaves is real but partial, so it is kept rather than deleted:
//! the [`tokenizer`] and [`config`] mapping are implemented and unit-tested
//! against the published `vocab.txt` and `config.json`, and the registry rows
//! carry `unsupported` explaining why they cannot load. Reading the weights
//! needs a legacy-pickle reader, tracked as `ferritin-goh.10`.

pub mod config;
pub mod tokenizer;

pub use config::{ProtBertConfig, protbert_config};
pub use tokenizer::{ProtBertTokenizer, UNKNOWN_RESIDUE};
