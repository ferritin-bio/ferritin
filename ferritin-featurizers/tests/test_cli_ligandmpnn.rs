//! These tests recreate the LigandMPNN Test Suite[found here](https://github.com/dauparas/LigandMPNN/blob/main/run_examples.sh)
//
// cargo flamegraph --bin ferritin-featurizers -- run --seed 111 --pdb-path ferritin-test-data/data/structures/1bc8.cif --model-type protein_mpnn --out-folder testout
// cargo instruments -t time --bin ferritin-featurizers -- run --seed 111 --pdb-path ferritin-test-data/data/structures/1bc8.cif --model-type protein_mpnn --out-folder testout
use assert_cmd::Command;
use ferritin_test_data::TestFile;
use std::path::Path;
use tempfile;
#[test]
fn test_cli_command_run_example_01() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--model-type")
        .arg("protein_mpnn")
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_02() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--temperature")
        .arg("0.05")
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_03() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_04() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--verbose")
        .arg("0")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}
#[test]
fn test_cli_command_run_example_05() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--save-stats")
        .arg("1");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_06() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--fixed-residues")
        .arg("C1 C2 C3 C4 C5 C6 C7 C8 C9 C10")
        .arg("--bias-AA")
        .arg("A:10.0");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_07() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--redesigned-residues")
        .arg("C1 C2 C3 C4 C5 C6 C7 C8 C9 C10")
        .arg("--bias-AA")
        .arg("A:10.0");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_08() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--batch-size")
        .arg("3")
        .arg("--number-of-batches")
        .arg("5");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_09() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--bias-AA")
        .arg("W:3.0,P:3.0,C:3.0,A:-3.0");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_10() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--bias-AA-per-residue")
        .arg("./inputs/bias_AA_per_residue.json");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_11() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--omit-AA")
        .arg("CDFGHILMNPQRSTVWY");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_12() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--omit-AA-per-residue")
        .arg("./inputs/omit_AA_per_residue.json");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_13() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--symmetry-residues")
        .arg("C1,C2,C3|C4,C5|C6,C7")
        .arg("--symmetry-weights")
        .arg("0.33,0.33,0.33|0.5,0.5|0.5,0.5");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_14() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--homo-oligomer")
        .arg("1")
        .arg("--number-of-batches")
        .arg("2");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}
#[test]
fn test_cli_command_run_example_15() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--file-ending")
        .arg("_xyz");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_16() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--zero-indexed")
        .arg("1")
        .arg("--number-of-batches")
        .arg("2");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_17() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--chains-to-design")
        .arg("A,B");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_18() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--parse-these-chains-only")
        .arg("A,B");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_19() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_20() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--checkpoint-ligand-mpnn")
        .arg("./model_params/ligandmpnn_v_32_005_25.pt")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_21() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--ligand-mpnn-use-atom-context")
        .arg("0");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_22() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--ligand-mpnn-use-side-chain-context")
        .arg("1")
        .arg("--fixed-residues")
        .arg("C1 C2 C3 C4 C5 C6 C7 C8 C9 C10");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_23() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("soluble_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_24() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("global_label_membrane_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--global-transmembrane-label")
        .arg("0");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_25() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("per_residue_label_membrane_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--transmembrane-buried")
        .arg("C1 C2 C3 C11")
        .arg("--transmembrane-interface")
        .arg("C4 C5 C6 C22");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_26() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--pdb-path")
        .arg(pdbfile)
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--fasta-seq-separation")
        .arg(":");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_27() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--pdb-path-multi")
        .arg("./inputs/pdb_ids.json")
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--seed")
        .arg("111");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_28() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--pdb-path-multi")
        .arg("./inputs/pdb_ids.json")
        .arg("--fixed-residues-multi")
        .arg("./inputs/fix_residues_multi.json")
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--seed")
        .arg("111");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_29() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--pdb-path-multi")
        .arg("./inputs/pdb_ids.json")
        .arg("--redesigned-residues-multi")
        .arg("./inputs/redesigned_residues_multi.json")
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--seed")
        .arg("111");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_30() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--pdb-path-multi")
        .arg("./inputs/pdb_ids.json")
        .arg("--omit-AA-per-residue-multi")
        .arg("./inputs/omit_AA_per_residue_multi.json")
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--seed")
        .arg("111");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_31() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--pdb-path-multi")
        .arg("./inputs/pdb_ids.json")
        .arg("--bias-AA-per-residue-multi")
        .arg("./inputs/bias_AA_per_residue_multi.json")
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--seed")
        .arg("111");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_32() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--model-type")
        .arg("ligand_mpnn")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg("./inputs/1BC8.pdb")
        .arg("--ligand-mpnn-cutoff-for-score")
        .arg("6.0")
        .arg("--out-folder")
        .arg(&out_folder);

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}

#[test]
fn test_cli_command_run_example_33() {
    let (pdbfile, _tmp) = TestFile::protein_03().create_temp().unwrap();
    let out_folder = tempfile::tempdir().unwrap().into_path();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("run")
        .arg("--seed")
        .arg("111")
        .arg("--pdb-path")
        .arg("./inputs/2GFB.pdb")
        .arg("--out-folder")
        .arg(&out_folder)
        .arg("--redesigned-residues")
        .arg("B82 B82A B82B B82C")
        .arg("--parse-these-chains-only")
        .arg("B");

    let assert = cmd.assert().success();
    println!("Successful command....");
    assert!(out_folder.exists());
    println!("Output: {:?}", assert.get_output());
}
