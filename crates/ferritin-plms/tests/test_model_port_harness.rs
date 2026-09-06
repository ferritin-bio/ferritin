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
    // All current ports have graduated to Stable; this test verifies that any
    // future Partial entries are explicitly registered (the registry is the
    // source of truth — do not delete this test).
    let partial_cases: Vec<_> = MODEL_PORT_CASES
        .iter()
        .filter(|case| case.status == PortStatus::Partial)
        .collect();
    // Stable ports must not regress to Partial without updating this list.
    for case in &partial_cases {
        assert_ne!(
            case.family, "esmc",
            "ESMC has graduated to Stable; remove it from Partial"
        );
        assert_ne!(
            case.family, "amplify",
            "AMPLIFY has graduated to Stable; remove it from Partial"
        );
        assert_ne!(
            case.family, "esm2",
            "ESM2 has graduated to Stable; remove it from Partial"
        );
    }
    // Partial cases are allowed to be empty when all ports are Stable.
    println!("{} partial port(s) tracked.", partial_cases.len());
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

/// Every port case must name a real registry model.
///
/// This is what keeps `MODEL_PORT_CASES` an annotation rather than a rival
/// inventory. Before ferritin-goh.1 there were two model lists that could
/// disagree; now the registry is canonical and this table only records what
/// coverage each entry has (ferritin-100.13).
#[test]
fn test_port_cases_reference_real_models() {
    use ferritin_plms::registry::lookup;

    for case in MODEL_PORT_CASES {
        let card = lookup(case.registry_id).unwrap_or_else(|| {
            panic!(
                "port case {}:{} names {:?}, which is not in REGISTRY",
                case.family, case.variant, case.registry_id
            )
        });
        assert_eq!(
            card.family_str(),
            case.family,
            "{}: port case and registry disagree on family",
            case.registry_id
        );
    }
}

/// A port case must not claim coverage for a model the registry says cannot
/// be loaded — that would advertise validation that cannot be running.
#[test]
fn test_no_coverage_claimed_for_unsupported_models() {
    use ferritin_plms::registry::lookup;

    for case in MODEL_PORT_CASES {
        let card = lookup(case.registry_id).expect("checked above");
        if !card.is_loadable() {
            assert!(
                !(case.has_remote_smoke || case.has_pytorch_checkpoint_test),
                "{}: registry says unsupported, so it cannot have live coverage",
                case.registry_id
            );
        }
    }
}
