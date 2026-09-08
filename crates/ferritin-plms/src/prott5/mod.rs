//! ProtT5 — the T5 encoder trained on UniRef50 (ferritin-goh.5).
//!
//! ProtT5 embeddings remain one of the most widely used protein
//! representations, and they give the crate a second, architecturally
//! independent embedding family to cross-check against ESM: a T5 encoder with
//! relative position buckets rather than a BERT-style stack with rotary or
//! learned absolute positions.
//!
//! # No architecture code
//!
//! `candle_transformers::models::t5` already provides `T5EncoderModel`, and
//! `candle-transformers` was already a workspace dependency, so this module is
//! a registry entry, a config mapping, and a tokenizer — nothing else. The
//! weight layout read off the published checkpoint lines up exactly with what
//! candle expects:
//!
//! | checkpoint key | candle |
//! |---|---|
//! | `shared.weight` | `T5EncoderModel::load` probes this first |
//! | `encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight` | `T5Attention` |
//! | `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight` | block 0 only |
//! | `encoder.block.{i}.layer.1.DenseReluDense.{wi,wo}.weight` | `T5DenseActDense` |
//! | `encoder.final_layer_norm.weight` | `T5Stack` |
//!
//! Note `q/k/v` are `[4096, 1024]`, not square: ProtT5-XL sets `d_kv: 128` with
//! `num_heads: 32`, so the attention inner dimension is 4096 against a
//! `d_model` of 1024. candle computes `inner_dim = num_heads * d_kv` and
//! handles that correctly.

pub mod prott5_runner;
pub mod tokenizer;
