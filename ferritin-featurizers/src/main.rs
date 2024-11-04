use clap::Parser;
mod cli;
mod commands;
// mod config;
// mod error;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    cli.execute()?;
    Ok(())
}
