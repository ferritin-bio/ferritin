//! Integration smoke tests for ESMFold2.
//!
//! Tests that require downloading model weights are marked `#[ignore]`.
//! Run them explicitly with:
//!
//! ```shell
//! cargo test -p ferritin-plms test_esmfold2 -- --nocapture
//! cargo test -p ferritin-plms test_esmfold2 -- --ignored --nocapture
//! ```

use anyhow::Result;
use candle_core::{Device, Tensor};
use ferritin_plms::{
    ESMFold2Config, ESMFold2Output,
    esmfold2::input_types::{
        DNAInput, LigandInput, Modification, ProteinInput, StructurePredictionInput,
    },
};

// ---------------------------------------------------------------------------
// Config tests
// ---------------------------------------------------------------------------

#[test]
fn test_esmfold2_config_fast_values() {
    let config = ESMFold2Config::fast();
    assert_eq!(config.d_single, 384);
    assert_eq!(config.d_pair, 256);
    assert_eq!(config.trunk_n_layers, 24);
    assert_eq!(config.lm_encoder_n_layers, 4);
    assert_eq!(config.lm_d_model, 2560);
    assert_eq!(config.inference_num_steps, 14);
    assert_eq!(config.num_plddt_bins, 50);
    assert_eq!(config.distogram_bins, 39);
    assert!(!config.msa_enabled, "Fast variant should have MSA disabled");
}

// ---------------------------------------------------------------------------
// Input construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_esmfold2_protein_input_valid() {
    let p = ProteinInput::new("A", "MKTAYIAKQRQISFVK").unwrap();
    assert_eq!(p.id, "A");
    assert_eq!(p.len(), 16);
    assert!(!p.is_empty());
    assert_eq!(format!("{p}"), "Protein[A](len=16)");
}

#[test]
fn test_esmfold2_protein_input_rejects_invalid() {
    assert!(
        ProteinInput::new("A", "").is_err(),
        "empty sequence must fail"
    );
    let err = ProteinInput::new("A", "MKT1YI").unwrap_err();
    assert!(err.contains("'1'"));
}

#[test]
fn test_esmfold2_dna_input_with_modification() {
    let dna = DNAInput::new("B", "ACGTACGT")
        .unwrap()
        .add_modification(Modification::new(3, "5MC"));
    assert_eq!(dna.len(), 8);
    assert_eq!(dna.modifications.len(), 1);
    assert_eq!(dna.modifications[0].ccd, "5MC");
    assert_eq!(dna.modifications[0].position, 3);
}

#[test]
fn test_esmfold2_ligand_input() {
    let l = LigandInput::from_ccd("L", "ATP");
    assert_eq!(l.id, "L");
    assert_eq!(l.ccd, vec!["ATP"]);
    assert_eq!(l.num_components(), 1);
}

#[test]
fn test_esmfold2_multi_chain_input() {
    let input = StructurePredictionInput::new()
        .add_protein(ProteinInput::new("A", "MKTAYIAK").unwrap())
        .add_dna(DNAInput::new("B", "ACGTACGT").unwrap())
        .add_ligand(LigandInput::from_ccd("L", "ATP"));

    assert_eq!(input.num_chains(), 3);
    assert_eq!(input.sequences[0].id(), "A");
    assert_eq!(input.sequences[1].id(), "B");
    assert_eq!(input.sequences[2].id(), "L");

    let s = format!("{input}");
    assert!(s.contains("Protein[A]"));
    assert!(s.contains("DNA[B]"));
    assert!(s.contains("Ligand[L]"));
}

// ---------------------------------------------------------------------------
// Output / mmCIF pipeline tests — no weights needed
// ---------------------------------------------------------------------------

fn make_ca_output(sequence: &str) -> Result<ESMFold2Output> {
    let l = sequence.len();
    let device = Device::Cpu;
    // Start at x=3.8 so residue 0 is never at the origin (mmCIF writer skips near-zero atoms).
    let ca_coords: Vec<f32> = (0..l)
        .flat_map(|i| [(i as f32 + 1.0) * 3.8, 0.0_f32, 1.0_f32])
        .collect();

    Ok(ESMFold2Output {
        sample_atom_coords: Tensor::from_vec(ca_coords, &[l, 3], &device)?,
        plddt: Tensor::from_vec(vec![0.85_f32; l], l, &device)?,
        ptm: Tensor::from_vec(vec![0.75_f32], 1_usize, &device)?,
        iptm: Tensor::from_vec(vec![0.75_f32], 1_usize, &device)?,
        pae: None,
        distogram_logits: None,
    })
}

#[test]
fn test_esmfold2_synthetic_output_to_mmcif() -> Result<()> {
    let sequence = "MKTAYIAK";
    let output = make_ca_output(sequence)?;
    let mmcif = output.to_mmcif(sequence, "A", "test_entry")?;

    assert!(mmcif.contains("data_test_entry"));
    assert!(mmcif.contains("loop_"));
    assert!(mmcif.contains("_atom_site.Cartn_x"));
    assert!(mmcif.contains("ATOM"));
    assert!(mmcif.contains("MET"), "M → MET");
    assert!(mmcif.contains("LYS"), "K → LYS");

    // Cα-only: exactly one ATOM per residue
    let atom_lines: Vec<&str> = mmcif.lines().filter(|l| l.starts_with("ATOM")).collect();
    assert_eq!(atom_lines.len(), sequence.len());

    // Every ATOM line must have 21 whitespace-separated fields
    for line in &atom_lines {
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 21, "field count wrong: {line}");
    }

    Ok(())
}

#[test]
fn test_esmfold2_mmcif_chain_id_embedded() -> Result<()> {
    let sequence = "ACDE";
    let output = make_ca_output(sequence)?;
    let mmcif = output.to_mmcif(sequence, "Z", "chain_test")?;

    for line in mmcif.lines().filter(|l| l.starts_with("ATOM")) {
        assert!(line.contains(" Z "), "chain id Z not found: {line}");
    }
    Ok(())
}

#[test]
fn test_esmfold2_mmcif_bfactor_from_plddt() -> Result<()> {
    // pLDDT = 0.85 → B-factor = 85.00
    let sequence = "MA";
    let output = make_ca_output(sequence)?;
    let mmcif = output.to_mmcif(sequence, "A", "bfac_test")?;

    // B-factor field is the 15th column (index 14) in each ATOM line
    for line in mmcif.lines().filter(|l| l.starts_with("ATOM")) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let bfac: f32 = fields[14].parse()?;
        assert!(
            (bfac - 85.0).abs() < 0.1,
            "B-factor should be ~85.00, got {bfac}: {line}"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Integration test (requires weights + forward-pass implementation)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires downloading ESMFold2-Fast weights (~755 MB) and forward pass implementation"]
fn test_esmfold2_fold_protein_integration() -> Result<()> {
    use ferritin_plms::{ESMFold2Models, ESMFold2Runner};

    let device = Device::Cpu;
    let runner = ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, device)?;

    let sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLKK";
    let output = runner.fold_protein(sequence, 3, 14)?;

    // pLDDT should be in [0, 1]
    let plddt_vals = output.plddt.to_vec1::<f32>()?;
    assert_eq!(plddt_vals.len(), sequence.len());
    for v in &plddt_vals {
        assert!(*v >= 0.0 && *v <= 1.0, "pLDDT out of range: {v}");
    }

    // Should produce valid mmCIF
    let mmcif = output.to_mmcif(sequence, "A", "integration_test")?;
    assert!(mmcif.contains("data_integration_test"));
    assert!(mmcif.lines().any(|l| l.starts_with("ATOM")));

    println!(
        "pLDDT mean: {:.3}",
        plddt_vals.iter().sum::<f32>() / plddt_vals.len() as f32
    );
    println!("mmCIF size: {} bytes", mmcif.len());
    Ok(())
}
