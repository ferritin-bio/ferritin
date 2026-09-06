// Shared across integration-test binaries; not every binary exercises every
// helper, so unused-in-this-binary items are expected.
#![allow(dead_code)]

use anyhow::{Result, anyhow};
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Error};
use candle_nn::VarBuilder;
use ferritin_plms::amplify::amplify::AMPLIFY;
use ferritin_plms::amplify::amplify_runner::{AmplifyModels, AmplifyRunner};
use ferritin_plms::esm2::esm2::ESM2;
use ferritin_plms::{ESM2Models, ESM2Runner, ProteinMPNN, ProteinMPNNConfig, device};
use ferritin_test_data::TestFile;

pub const TEST_SEQUENCE: &str = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";
pub const HAMMING_CUTOFF: f32 = 0.7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortStatus {
    Stable,
    Partial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPortCase {
    pub family: &'static str,
    pub variant: &'static str,
    pub source_artifact: &'static str,
    pub rust_backend: &'static str,
    pub status: PortStatus,
    pub has_local_tokenizer: bool,
    pub has_remote_smoke: bool,
    pub has_pytorch_checkpoint_test: bool,
    pub notes: &'static str,
}

pub const MODEL_PORT_CASES: &[ModelPortCase] = &[
    ModelPortCase {
        family: "amplify",
        variant: "AMP120M",
        source_artifact: "huggingface safetensors",
        rust_backend: "candle",
        status: PortStatus::Stable,
        has_local_tokenizer: true,
        has_remote_smoke: true,
        has_pytorch_checkpoint_test: false,
        notes: "Candle runner loads config/tokenizer/weights from Hugging Face.",
    },
    ModelPortCase {
        family: "amplify",
        variant: "AMP350M",
        source_artifact: "huggingface safetensors",
        rust_backend: "candle",
        status: PortStatus::Stable,
        has_local_tokenizer: true,
        has_remote_smoke: false,
        has_pytorch_checkpoint_test: false,
        notes: "Runner exists, but the automated smoke coverage is still focused on AMP120M.",
    },
    ModelPortCase {
        family: "esm2",
        variant: "T6_8M",
        source_artifact: "huggingface safetensors",
        rust_backend: "candle",
        status: PortStatus::Stable,
        has_local_tokenizer: true,
        has_remote_smoke: true,
        has_pytorch_checkpoint_test: false,
        notes: "Best lightweight smoke target for the ESM2 Candle port.",
    },
    ModelPortCase {
        family: "esm2",
        variant: "T30_150M",
        source_artifact: "huggingface safetensors",
        rust_backend: "candle",
        status: PortStatus::Stable,
        has_local_tokenizer: true,
        has_remote_smoke: true,
        has_pytorch_checkpoint_test: false,
        notes: "Higher-capacity ESM2 variant used for integration validation.",
    },
    ModelPortCase {
        family: "proteinmpnn",
        variant: "v48_020",
        source_artifact: "embedded pytorch checkpoint",
        rust_backend: "candle",
        status: PortStatus::Stable,
        has_local_tokenizer: false,
        has_remote_smoke: false,
        has_pytorch_checkpoint_test: true,
        notes: "PyTorch checkpoint loading is validated against embedded test data.",
    },
    ModelPortCase {
        family: "esmc",
        variant: "ESMC-300M",
        source_artifact: "huggingface safetensors (biohub/ESMC-300M)",
        rust_backend: "candle",
        status: PortStatus::Stable,
        has_local_tokenizer: false,
        has_remote_smoke: true,
        has_pytorch_checkpoint_test: false,
        notes: "ESMCRunner loads weights from biohub/ESMC-300M via hf_hub. Auto-detects ESMCForMaskedLM prefix. 600M and 6B configs also available.",
    },
];

pub fn hamming_ratio(s1: &str, s2: &str) -> f32 {
    let matches = s1
        .chars()
        .zip(s2.chars())
        .filter(|(c1, c2)| c1 == c2)
        .count();
    matches as f32 / s1.len() as f32
}

pub fn validate_local_sequence_tokenizers() -> Result<()> {
    let amplify = AMPLIFY::load_tokenizer()?;
    let amplify_tokens = amplify
        .encode(TEST_SEQUENCE, false)
        .map_err(|e| anyhow!("AMPLIFY tokenizer failed: {e}"))?;
    if amplify_tokens.len() != TEST_SEQUENCE.len() {
        return Err(anyhow!(
            "AMPLIFY tokenizer length mismatch: expected {}, got {}",
            TEST_SEQUENCE.len(),
            amplify_tokens.len()
        ));
    }

    let esm2 = ESM2::load_tokenizer()?;
    let esm2_tokens = esm2
        .encode("MLKLRV", false)
        .map_err(|e| anyhow!("ESM2 tokenizer failed: {e}"))?;
    if esm2_tokens.get_tokens() != ["M", "L", "K", "L", "R", "V"] {
        return Err(anyhow!("ESM2 tokenizer emitted an unexpected token stream"));
    }

    Ok(())
}

pub fn validate_proteinmpnn_checkpoint() -> Result<(), Error> {
    let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
    let pth = PthTensors::new(mpnn_file, Some("model_state_dict"))?;
    let vb = VarBuilder::from_backend(Box::new(pth), DType::F32, Device::Cpu);
    let config = ProteinMPNNConfig::proteinmpnn();
    ProteinMPNN::load(vb, &config)?;
    Ok(())
}

pub fn run_remote_esm2_smoke(model: ESM2Models) -> Result<()> {
    let esm2 = ESM2Runner::from_pretrained(model, device(false)?)?;
    let output = esm2.run_forward(TEST_SEQUENCE)?;
    let output_sequence = esm2.decode_logits(output)?;
    let hamming = hamming_ratio(TEST_SEQUENCE, &output_sequence);
    if hamming <= HAMMING_CUTOFF {
        return Err(anyhow!(
            "ESM2 reconstruction quality below cutoff: {hamming:.3} <= {HAMMING_CUTOFF:.3}"
        ));
    }
    Ok(())
}

pub fn run_remote_amplify_prediction_smoke(model: AmplifyModels) -> Result<()> {
    let amplify = AmplifyRunner::from_pretrained(model, device(false)?)?;
    let prediction = amplify
        .get_best_prediction(TEST_SEQUENCE)
        .map_err(|err| anyhow!("AMPLIFY prediction failed: {err}"))?;
    if prediction.is_empty() {
        return Err(anyhow!("AMPLIFY returned an empty prediction"));
    }
    Ok(())
}

pub fn stable_cases() -> impl Iterator<Item = &'static ModelPortCase> {
    MODEL_PORT_CASES
        .iter()
        .filter(|case| case.status == PortStatus::Stable)
}
