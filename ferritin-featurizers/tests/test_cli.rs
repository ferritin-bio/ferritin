use assert_cmd::Command;
use ferritin_test_data::TestFile;
use tempfile;

#[test]
fn test_cli_command() {
    // input and output tempfiles
    let (ciffile, _tmp) = TestFile::protein_01().create_temp().unwrap();
    let tempfile = tempfile::NamedTempFile::new().unwrap();

    let tempfile = "test.safetensor";
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();

    cmd.arg("featurize")
        .arg("--input")
        .arg(ciffile)
        .arg("--output")
        // .arg(tempfile.path())
        .arg(tempfile);

    cmd.assert().success();
}
