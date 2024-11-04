use super::commands;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Featurize {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        output: String,
    },
}

impl Cli {
    pub fn execute(self) -> anyhow::Result<()> {
        match self.command {
            Commands::Featurize { input, output } => commands::featurize::execute(input, output),
        }
    }
}
