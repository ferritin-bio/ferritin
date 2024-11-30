use assert_cmd::Command;
use ferritin_test_data::TestFile;
use std::path::Path;
use tempfile;

#[test]
fn test_cli_command_featurize() {
    let (ciffile, _tmp) = TestFile::protein_01().create_temp().unwrap();
    let tempfile = tempfile::NamedTempFile::new().unwrap();
    let outfile = tempfile.path();
    let outpath = Path::new(&outfile);

    // let outfile = "test.safetensors".to_string();
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("featurize")
        .arg("--input")
        .arg(ciffile)
        .arg("--output")
        .arg(&outfile);

    // Actually execute the command and verify success
    // and test that the file is of non-zero-size
    cmd.assert().success();

    assert!(outpath.exists());
    assert!(outpath.metadata().unwrap().len() > 0);
}

#[test]
#[ignore]
fn test_cli_command_run_example_01() {
    // see: https://github.com/dauparas/LigandMPNN/blob/main/run_examples.sh
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

    // Actually execute the command and verify success
    // and test that the file is of non-zero-size
    let assert = cmd.assert().success();

    println!("Successful command....");
    assert!(out_folder.exists());

    // Print the output from the assertion
    println!("Output: {:?}", assert.get_output());
}
