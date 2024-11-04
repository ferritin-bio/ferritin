// src/commands/command1.rs
pub fn execute(name: String) -> anyhow::Result<()> {
    println!("Executing command1 with name: {}", name);
    Ok(())
}
