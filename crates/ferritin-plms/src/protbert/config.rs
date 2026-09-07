//! Mapping ProtBert's `config.json` onto candle's BERT config.
//!
//! The two do not line up directly. `Rostlab/prot_bert/config.json` omits
//! three fields that `candle_transformers::models::bert::Config` requires —
//! `layer_norm_eps`, `pad_token_id` and `model_type` — so deserializing it
//! straight into candle's struct fails. Rather than patch the JSON, this
//! module parses the published shape and fills the gaps with HuggingFace's
//! documented `BertConfig` defaults, which is what `transformers` does when it
//! loads the same file.

use candle_transformers::models::bert::{Config as BertConfig, HiddenAct, PositionEmbeddingType};
use serde::Deserialize;

/// HuggingFace `BertConfig`'s default layer-norm epsilon.
///
/// Not stated in either Rostlab `config.json`, so `transformers` supplies it.
/// Recorded as a named constant because a silently different epsilon changes
/// every layer's output slightly and would be invisible in a diff.
const DEFAULT_LAYER_NORM_EPS: f64 = 1e-12;

/// `[PAD]` is line 0 of ProtBert's `vocab.txt`.
const PAD_TOKEN_ID: usize = 0;

/// ProtBert's `config.json`, as published.
///
/// Only the fields the checkpoint actually declares. Anything absent is
/// supplied by [`to_candle`][Self::to_candle] rather than defaulted here, so
/// the distinction between "the file said this" and "we chose this" stays
/// visible.
#[derive(Debug, Clone, Deserialize)]
pub struct ProtBertConfig {
    /// Hidden width.
    pub hidden_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Attention heads per block.
    pub num_attention_heads: usize,
    /// Feed-forward inner width.
    pub intermediate_size: usize,
    /// Activation; `"gelu"` in both repos.
    pub hidden_act: HiddenAct,
    /// Dropout probability; 0.0 in both repos, and unused at inference.
    pub hidden_dropout_prob: f64,
    /// Size of the learned absolute position table — 40000 here, which is
    /// unusually large and accounts for ~164 MB of the checkpoint.
    pub max_position_embeddings: usize,
    /// Segment-embedding vocabulary; 2, as in stock BERT.
    pub type_vocab_size: usize,
    /// Weight-init range. Unused at inference, kept for fidelity.
    pub initializer_range: f64,
    /// Alphabet size: 5 specials plus 25 residue symbols.
    pub vocab_size: usize,
}

impl ProtBertConfig {
    /// Convert to the config candle's BERT implementation expects.
    pub fn to_candle(&self) -> BertConfig {
        BertConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            hidden_act: self.hidden_act,
            hidden_dropout_prob: self.hidden_dropout_prob,
            max_position_embeddings: self.max_position_embeddings,
            type_vocab_size: self.type_vocab_size,
            initializer_range: self.initializer_range,
            // Supplied, not published — see the constants above.
            layer_norm_eps: DEFAULT_LAYER_NORM_EPS,
            pad_token_id: PAD_TOKEN_ID,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            // `BertModel::load` uses this only as a fallback weight prefix.
            // The checkpoint is a `BertForMaskedLM`, so its parameters live
            // under `bert.`, which `BertForMaskedLM::load` handles directly.
            model_type: Some("bert".to_string()),
        }
    }
}

/// The config shared by `prot_bert` and `prot_bert_bfd`.
///
/// Both repos publish byte-identical `config.json` files, so one built-in
/// config covers them. It exists as a fallback for the same reason ESM-2 has
/// one — but note that the ESM-2 loader's silent `unwrap_or(fallback)` was a
/// real bug (ferritin-goh.9), so any caller using this must be loud about it.
pub fn protbert_config() -> ProtBertConfig {
    ProtBertConfig {
        hidden_size: 1024,
        num_hidden_layers: 30,
        num_attention_heads: 16,
        intermediate_size: 4096,
        hidden_act: HiddenAct::Gelu,
        hidden_dropout_prob: 0.0,
        max_position_embeddings: 40000,
        type_vocab_size: 2,
        initializer_range: 0.02,
        vocab_size: 30,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact contents of `Rostlab/prot_bert/config.json`, and of
    /// `Rostlab/prot_bert_bfd/config.json`, which is byte-identical.
    const PUBLISHED: &str = r#"{
      "architectures": ["BertForMaskedLM"],
      "attention_probs_dropout_prob": 0.0,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.0,
      "hidden_size": 1024,
      "initializer_range": 0.02,
      "intermediate_size": 4096,
      "max_position_embeddings": 40000,
      "num_attention_heads": 16,
      "num_hidden_layers": 30,
      "type_vocab_size": 2,
      "vocab_size": 30
    }"#;

    /// The published file parses. This is the check that would have caught the
    /// mismatch had it been written before the config mapping.
    #[test]
    fn test_published_config_parses() {
        let c: ProtBertConfig = serde_json::from_str(PUBLISHED).unwrap();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_hidden_layers, 30);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.vocab_size, 30);
        assert_eq!(c.max_position_embeddings, 40000);
    }

    /// candle's own `Config` cannot read the published file: it requires
    /// `layer_norm_eps` and `pad_token_id`, which Rostlab omits. This is the
    /// entire reason [`ProtBertConfig`] exists, so pin it — if a future candle
    /// gives those fields defaults, this test fails and the wrapper can go.
    #[test]
    fn test_candle_config_cannot_read_the_published_file() {
        let direct: Result<BertConfig, _> = serde_json::from_str(PUBLISHED);
        assert!(
            direct.is_err(),
            "candle's Config parsed the published config.json; the ProtBertConfig \
             wrapper may no longer be needed"
        );
    }

    #[test]
    fn test_built_in_config_matches_the_published_one() {
        let published: ProtBertConfig = serde_json::from_str(PUBLISHED).unwrap();
        let built_in = protbert_config();
        let (a, b) = (published.to_candle(), built_in.to_candle());
        assert_eq!(a, b, "the built-in fallback has drifted from config.json");
    }

    #[test]
    fn test_defaults_are_supplied_for_the_omitted_fields() {
        let c = protbert_config().to_candle();
        assert_eq!(c.layer_norm_eps, DEFAULT_LAYER_NORM_EPS);
        assert_eq!(c.pad_token_id, PAD_TOKEN_ID, "[PAD] is line 0 of vocab.txt");
        assert_eq!(c.position_embedding_type, PositionEmbeddingType::Absolute);
    }
}
