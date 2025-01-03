use anyhow::{Error as E, Result};
use candle_core::DType;
use candle_examples::device;
use clap::Parser;
use ferritin_plms::{AmplifyModels, AmplifyRunner};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value = "false")]
    cpu: bool,

    /// Which AMPLIFY Model to use, either '120M' or '350M'.
    #[arg(long, value_parser = ["120M", "350M"], default_value = "120M")]
    model_id: String,

    /// Protein String
    #[arg(long)]
    protein_string: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;

    let amp_model = match args.model_id.as_str() {
        "120M" => AmplifyModels::AMP120M,
        "350M" => AmplifyModels::AMP350M,
        &_ => panic!("Only 2 options"),
    };
    let amprunner = AmplifyRunner::load_model(amp_model, device)?;
    let prot_sequence = args.protein_string.unwrap();

    // Runs the model and returns the full, manipulateable result
    let outputs = amprunner.run_forward(&prot_sequence);
    // Runs the model and returns the top hit from each logit
    let top_hit = amprunner.get_best_prediction(&prot_sequence);
    // Runs the model and returns the top probabilities
    let get_probabilities = amprunner.get_pseudo_probabilities(&prot_sequence);
    // Runs the model and returns the contactmap
    let contact_map = amprunner.get_contact_map(&prot_sequence);

    Ok(())
}
