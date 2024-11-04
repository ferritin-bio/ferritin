use assert_cmd::Command;

#[test]
fn test_cli_command() {
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();
    cmd.arg("featurize")
        .arg("--input")
        .arg("test")
        .arg("--output")
        .arg("test");

    cmd.assert().success();
}
