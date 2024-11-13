use anyhow::Result;
use candle_core::{DType, Device, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use ferritin_featurizers::{AMPLIFYConfig, ProteinTokenizer, AMPLIFY};
use safetensors::SafeTensors;

fn main() -> Result<()> {
    let model_id = "chandar-lab/AMPLIFY_120M";
    let revision = "main";
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let weights_path = repo.get("model.safetensors")?;

    // Available for printing Tensor data....
    let print_tensor_info = false;
    if print_tensor_info {
        println!("Model tensors:");
        let weights = std::fs::read(&weights_path)?;
        let tensors = SafeTensors::deserialize(&weights)?;
        tensors.names().iter().for_each(|tensor_name| {
            if let Ok(tensor_info) = tensors.tensor(tensor_name) {
                println!(
                    "Tensor: {:<44}  ||  Shape: {:?}",
                    tensor_name,
                    tensor_info.shape(),
                );
            }
        });
    }

    println!("Loading the Amplify Model ......");
    // https://github.com/huggingface/candle/blob/main/candle-examples/examples/clip/main.rs#L91C1-L92C101
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &Device::Cpu)?
    };
    let config = AMPLIFYConfig::amp_120m();
    let model = AMPLIFY::load(vb, &config)?;

    println!("Tokenizing and Modelling a Sequence from Swissprot...");
    let tokenizer = repo.get("tokenizer.json")?;
    let protein_tokenizer = ProteinTokenizer::new(tokenizer)?;
    let sprot_01 = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";
    let pmatrix = protein_tokenizer.encode(&[sprot_01.to_string()], None, false, false)?;
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case
    let encoded = model.forward(&pmatrix, None, false, false)?;

    println!("Assessing the Predictions.......");
    // As of Nov 13 this is definitely not right....
    // Input:   MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL
    // Output:  MSVQLNIVGQSAAWTHGAAVCATCAQTFWPMSRGRQPPVNMSRFTARCTECIWYEAAFNARFNFVHLYNCGPNMSECLANMSWWYACQFGVHMSKSHYCGNKPLGTDNTKMMHHRECTSTVVWKHWPLCKVTVCYRHGLVSCTMHQRSTWTPRNEASWVPEWETSTPEHTCGDYWACQMPAGHGVCCCMMTEHWKPHTRVVCQTIEMWTYLQTYYYFWGVPEPCHHHIWTEPMPTSTSTSYDVVMYTTSGFGQHHW
    let predictions = encoded.logits.argmax(D::Minus1)?;
    let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
    let decoded = protein_tokenizer.decode(indices.as_slice(), true)?;
    println!("Encoded Logits Dimension: {:?}, ", encoded.logits);
    println!("indices: {:?}", indices);
    println!("Decoded Values: {}", decoded.replace(" ", ""));
    Ok(())
}
