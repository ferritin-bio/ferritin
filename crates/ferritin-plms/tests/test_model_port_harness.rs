mod support;

use anyhow::Result;
use support::model_harness::{
    MODEL_PORT_CASES, PortStatus, stable_cases, validate_local_sequence_tokenizers,
    validate_proteinmpnn_checkpoint,
};

#[test]
fn test_model_port_registry_tracks_current_families() {
    let families: Vec<_> = MODEL_PORT_CASES.iter().map(|case| case.family).collect();
    assert!(families.contains(&"amplify"));
    assert!(families.contains(&"esm2"));
    assert!(families.contains(&"proteinmpnn"));
    assert!(families.contains(&"esmc"));
}

#[test]
fn test_stable_ports_have_an_automated_validation_path() {
    for case in stable_cases() {
        assert!(
            case.has_local_tokenizer || case.has_remote_smoke || case.has_pytorch_checkpoint_test,
            "stable port {}:{} needs at least one automated validation path",
            case.family,
            case.variant
        );
    }
}

#[test]
fn test_partial_ports_are_called_out_explicitly() {
    let partial_cases: Vec<_> = MODEL_PORT_CASES
        .iter()
        .filter(|case| case.status == PortStatus::Partial)
        .collect();
    assert!(!partial_cases.is_empty());
    assert!(partial_cases.iter().any(|case| case.family == "esmc"));
}

#[test]
fn test_local_sequence_tokenizer_contracts() -> Result<()> {
    validate_local_sequence_tokenizers()
}

#[test]
fn test_proteinmpnn_checkpoint_contract() -> Result<()> {
    validate_proteinmpnn_checkpoint()?;
    Ok(())
}
