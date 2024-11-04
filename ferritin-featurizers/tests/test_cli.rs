use assert_cmd::Command;
use ferritin_test_data::TestFile;
use std::path::Path;
use tempfile;

#[test]
fn test_cli_command() {
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
