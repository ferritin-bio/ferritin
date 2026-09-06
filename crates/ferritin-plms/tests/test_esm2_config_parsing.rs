//! `ESM2Config` must parse the config.json files real ESM-family models ship
//! (ferritin-goh.9).
//!
//! The struct used to demand ~23 non-Option fields, several of which real
//! configs omit. That alone would be a papercut; what made it a bug is where
//! the failure went:
//!
//! ```ignore
//! serde_json::from_str::<ESM2Config>(&config_str).unwrap_or(fallback_config)
//! ```
//!
//! A missing field failed deserialization, `unwrap_or` discarded the error
//! without a word, and the **hardcoded config for a different model** was used
//! instead. For `SaProt_35M_AF2` that means a 446-token vocabulary model
//! loading with a 33-token config. The VarBuilder would likely have caught
//! that particular case on the embedding shape — but any divergence that
//! happened to be shape-compatible would have loaded cleanly and produced
//! wrong numbers.
//!
//! These fixtures are the published `config.json` of each model, committed
//! verbatim under `tests/fixtures/configs/`, so this is checked against what
//! HuggingFace actually serves rather than a hand-written approximation. Add a
//! fixture here whenever a new ESM-family model is registered — that is what
//! stops this recurring.

use ferritin_plms::ESM2Config;
use std::path::PathBuf;

fn config_json(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/configs")
        .join(format!("{name}.json"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing config fixture {}: {e}", path.display()))
}

fn parse(name: &str) -> ESM2Config {
    serde_json::from_str::<ESM2Config>(&config_json(name))
        .unwrap_or_else(|e| panic!("{name}'s published config.json must deserialize: {e}"))
}

/// The baseline: a stock ESM-2 config with every field present.
#[test]
fn test_parses_esm2_t6_8m() {
    let config = parse("esm2_t6_8M_UR50D");
    assert_eq!(config.hidden_size, 320);
    assert_eq!(config.num_hidden_layers, 6);
    assert_eq!(config.num_attention_heads, 20);
    assert_eq!(config.vocab_size, 33);
    assert_eq!(config.position_embedding_type, "rotary");
}

/// ESM-1v shares the schema but uses absolute position embeddings — a field
/// that must stay required, since defaulting it would silently change the
/// attention math.
#[test]
fn test_parses_esm1v_t33_650m() {
    let config = parse("esm1v_t33_650M_UR90S_1");
    assert_eq!(config.hidden_size, 1280);
    assert_eq!(config.num_hidden_layers, 33);
    assert_eq!(config.position_embedding_type, "absolute");
}

/// SaProt omits `is_folding_model`, `esmfold_config` and `vocab_list`
/// entirely. This is the config that used to fail to parse and get silently
/// replaced by ESM-2's.
#[test]
fn test_parses_saprot_35m_which_omits_fields() {
    let raw = config_json("SaProt_35M_AF2");
    for absent in ["is_folding_model", "esmfold_config", "vocab_list"] {
        assert!(
            !raw.contains(absent),
            "fixture should still be the published config, which omits {absent}"
        );
    }

    let config = parse("SaProt_35M_AF2");
    assert_eq!(config.hidden_size, 480);
    assert_eq!(config.num_hidden_layers, 12);

    // The values that made the silent fallback dangerous: nothing like ESM-2's.
    assert_eq!(config.vocab_size, 446, "SaProt has a structure-aware vocab");
    assert_eq!(config.mask_token_id, 4);

    // Omitted fields take their defaults rather than failing the parse.
    assert!(!config.is_folding_model);
    assert_eq!(config.esmfold_config, None);
    assert_eq!(config.vocab_list, None);
}

/// A config missing a shape-critical field must FAIL rather than default.
///
/// This is the other half of the fix: `#[serde(default)]` was applied only to
/// fields the forward pass never reads. Defaulting `hidden_size` to 0 would
/// trade a loud parse error for a confusing shape error — or worse, silence.
#[test]
fn test_missing_required_field_is_an_error() {
    let mut value: serde_json::Value =
        serde_json::from_str(&config_json("esm2_t6_8M_UR50D")).unwrap();
    value.as_object_mut().unwrap().remove("hidden_size");

    let err = serde_json::from_value::<ESM2Config>(value)
        .map(|_| ())
        .expect_err("a config without hidden_size must not deserialize");
    assert!(
        err.to_string().contains("hidden_size"),
        "the error should name the missing field; got: {err}"
    );
}

/// Likewise for a field that changes the math rather than the shapes.
#[test]
fn test_missing_position_embedding_type_is_an_error() {
    let mut value: serde_json::Value =
        serde_json::from_str(&config_json("esm2_t6_8M_UR50D")).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("position_embedding_type");

    serde_json::from_value::<ESM2Config>(value)
        .map(|_| ())
        .expect_err("position_embedding_type must stay required: rotary vs absolute");
}

/// Unknown fields are ignored, so a newer `transformers` version adding keys
/// does not break loading.
#[test]
fn test_unknown_fields_are_ignored() {
    let mut value: serde_json::Value =
        serde_json::from_str(&config_json("esm2_t6_8M_UR50D")).unwrap();
    value.as_object_mut().unwrap().insert(
        "some_future_field".to_string(),
        serde_json::Value::Bool(true),
    );

    serde_json::from_value::<ESM2Config>(value).expect("unknown fields should be ignored");
}
