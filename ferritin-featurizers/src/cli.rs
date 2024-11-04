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
    Command1 {
        #[arg(short, long)]
        name: String,
    },
}

impl Cli {
    pub fn execute(self) -> anyhow::Result<()> {
        match self.command {
            Commands::Command1 { name } => commands::featurize::execute(name),
        }
    }
}
