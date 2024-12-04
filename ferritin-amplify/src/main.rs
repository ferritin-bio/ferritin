//! Entrypoint for CLI

use clap::Parser;
mod cli;
mod models;
pub mod amplify;
// use models::ligandmpnn::featurizer::LMPNNFeatures;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    cli.execute()?;
    Ok(())
}
