use anyhow::Result;
use clap::Parser;
use ferritin_plms::{AmplifyModels, AmplifyOutput, AmplifyRunner, device};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
    /// Which AMPLIFY Model to use, either '120M' or '350M'.
    #[arg(long, value_parser = ["120M", "350M"], default_value = "120M")]
    model_id: String,
    /// Protein sequence to evaluate. Defaults to a short demo sequence.
    #[arg(long)]
    protein_string: Option<String>,
}

// A short real protein sequence (Ubiquitin, human) used as the default demo.
const DEFAULT_SEQUENCE: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

fn main() -> Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;
    let amp_model = match args.model_id.as_str() {
        "120M" => AmplifyModels::AMP120M,
        "350M" => AmplifyModels::AMP350M,
        &_ => panic!("Only 2 options"),
    };
    let amprunner = AmplifyRunner::load_model(amp_model, device)?;
    let prot_sequence = args.protein_string.unwrap_or_else(|| {
        println!("No --protein-string provided, using default demo sequence.");
        DEFAULT_SEQUENCE.to_string()
    });
    // Runs the model and returns the full, manipulateable result
    let _outputs = amprunner.run_forward(&prot_sequence);
    // println!("Outputs: {:?}", outputs);
    // Runs the model and returns the top hit from each logit
    let top_hit = amprunner.get_best_prediction(&prot_sequence);
    println!("Top_hit: {:?}", top_hit);
    // Runs the model and returns the top probabilities
    let _pseudo_probabilities = amprunner.get_pseudo_probabilities(&prot_sequence);
    // Runs the model and returns the contactmap
    let _contact_map = amprunner.get_contact_map(&prot_sequence);
    Ok(())
}
