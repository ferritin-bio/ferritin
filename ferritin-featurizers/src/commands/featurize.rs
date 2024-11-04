// src/commands/command1.rs
pub fn execute(input: String, output: String) -> anyhow::Result<()> {
    println!("Executing command1 with name: {}", input);
    println!("Executing command1 with name: {}", output);
    Ok(())
}
