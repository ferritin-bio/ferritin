use assert_cmd::Command;

#[test]
fn test_cli_command() {
    let mut cmd = Command::cargo_bin("ferritin-featurizers").unwrap();
    cmd.arg("command1").arg("--name").arg("test");

    cmd.assert().success();
}
