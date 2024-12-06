use anyhow::Result;
use candle_core::{DType, Device, D};
use candle_examples::device;
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use ferritin_amplify::{AMPLIFYConfig, ProteinTokenizer, AMPLIFY};
use safetensors::SafeTensors;

fn main() -> Result<()> {
    println!("Loading the Amplify Model ......");
    #[cfg(target_os = "macos")]
    let use_gpu = false;
    #[cfg(not(target_os = "macos"))]
    let use_gpu = true;

    let dev = device(use_gpu)?;
    let (tokenizer, amplify) = AMPLIFY::load_from_huggingface(dev.clone())?;
    let sprot_01 = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";
    let pmatrix = tokenizer.encode(&[sprot_01.to_string()], None, false, false)?;
    println!("Pmatrix Device: {:?}", pmatrix.device());
    let pmatrix = pmatrix.to_device(&dev)?;
    println!("Pmatrix Device: {:?}", pmatrix.device());
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case
    let encoded = amplify.forward(&pmatrix, None, false, false)?;
    println!("Assessing the Predictions.......");
    // As of Nov 13 this is definitely not right....
    // Input:   MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL
    // Output:  MSVQLNIVGQSAAWTHGAAVCATCAQTFWPMSRGRQPPVNMSRFTARCTECIWYEAAFNARFNFVHLYNCGPNMSECLANMSWWYACQFGVHMSKSHYCGNKPLGTDNTKMMHHRECTSTVVWKHWPLCKVTVCYRHGLVSCTMHQRSTWTPRNEASWVPEWETSTPEHTCGDYWACQMPAGHGVCCCMMTEHWKPHTRVVCQTIEMWTYLQTYYYFWGVPEPCHHHIWTEPMPTSTSTSYDVVMYTTSGFGQHHW
    let predictions = encoded.logits.argmax(D::Minus1)?;
    let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
    let decoded = tokenizer.decode(indices.as_slice(), true)?;
    println!("Encoded Logits Dimension: {:?}, ", encoded.logits);
    println!("indices: {:?}", indices);
    println!("Decoded Values: {}", decoded.replace(" ", ""));
    Ok(())
}
