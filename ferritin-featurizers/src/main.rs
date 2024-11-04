use clap::Parser;
mod cli;
mod commands;
mod models;

use models::ligandmpnn::featurizer::LMPNNFeatures;

// mod config;
// mod error;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    cli.execute()?;
    Ok(())
}
