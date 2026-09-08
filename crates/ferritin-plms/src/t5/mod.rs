//! The T5 encoder family: ProtT5 and Ankh (ferritin-goh.5, ferritin-goh.6).
//!
//! ProtT5 embeddings remain one of the most widely used protein
//! representations, and they give the crate a second, architecturally
//! independent embedding family to cross-check against ESM: a T5 encoder with
//! relative position buckets rather than a BERT-style stack with rotary or
//! learned absolute positions. Ankh is a third, arriving essentially for free.
//!
//! # One runner for both
//!
//! ProtT5 and Ankh differ in nothing this module cares about. Their residue
//! alphabets are the *same letters at the same ids* — `A` = 3, `L` = 4, …
//! `Z` = 27, frequency-ordered — and both wrap a sequence in a trailing `</s>`
//! and no BOS. So [`tokenizer`] serves both, and the differences that remain
//! come out of each checkpoint's own `config.json`:
//!
//! | | ProtT5-XL | Ankh-base | Ankh-large |
//! |---|---|---|---|
//! | `d_model` | 1024 | 768 | 1536 |
//! | encoder layers | 24 | 48 | 48 |
//! | `feed_forward_proj` | `relu` | `gated-gelu` | `gated-gelu` |
//! | published dtype | F16 | F32 | F32 |
//!
//! The gated FFN is the one substantive difference: candle loads it as
//! `T5DenseGatedActDense` rather than the plain dense ProtT5 uses.
//!
//! Ankh ships its own `tokenizer.json` (a Unigram vocabulary with no
//! SentencePiece boundary marker, so it takes bare sequences), and the runner
//! deliberately does not use it. That reuse is *checked*, not assumed: the
//! parity fixture carries HuggingFace's own token ids for both models, so a
//! divergence fails on the ids before it reaches the embeddings.
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
//! handles that correctly. Ankh-large is the same story in the other
//! direction: `d_kv: 64` and `num_heads: 16` give an inner dimension of 1024
//! against a `d_model` of 1536.

pub mod runner;
pub mod tokenizer;
