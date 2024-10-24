use clap::Parser;
use pseutils::pymolparsing::psedata::PSEData;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the PSE file
    #[arg(short, long)]
    psefile: String,

    /// Output directory
    #[arg(short, long)]
    outputdir: String,
}

fn main() {
    let args = Args::parse();

    println!("PSE file: {}", args.psefile);
    println!("Output directory: {}", args.outputdir);

    let psedata: PSEData = PSEData::load(&args.psefile).expect("Reachable PSE file");
    let _ = psedata.to_disk_full(&args.outputdir);
}
