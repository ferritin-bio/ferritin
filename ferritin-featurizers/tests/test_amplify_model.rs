use anyhow::{Error, Result};
use candle_core::{DType, Device, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use ferritin_featurizers::{AMPLIFYConfig, ProteinTokenizer, AMPLIFY};

#[test]
fn test_amplify_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "chandar-lab/AMPLIFY_120M";
    let revision = "main";
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    // Load and analyze the safetensors file
    let weights_path = repo.get("model.safetensors")?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &Device::Cpu)?
    };
    let config = AMPLIFYConfig::amp_120m();
    let model = AMPLIFY::load(vb, &config)?;
    let tokenizer = repo.get("tokenizer.json")?;
    let protein_tokenizer = ProteinTokenizer::new(tokenizer)?;
    // let sprot_01 = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";
    let AMPLIFY_TEST_SEQ = "MSVVGIDLGFQSCYVAVARAGGIETIANEYSDRCTPACISFGPKNR";
    let pmatrix = protein_tokenizer.encode(&[AMPLIFY_TEST_SEQ.to_string()], None, true, false)?;
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case
    let encoded = model.forward(&pmatrix, None, true, true)?;
    // Choosing ARGMAX. We expect this to be the most predicted sequence.
    // it should return the identity of an unmasked sequence
    let predictions = &encoded.logits.argmax(D::Minus1)?;
    let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
    let decoded = protein_tokenizer.decode(indices.as_slice(), true)?;
    let final_seq = format!("S{}S", AMPLIFY_TEST_SEQ);
    assert_eq!(final_seq, decoded.replace(" ", ""));
    assert!(&encoded.attentions.is_some());
    assert!(&encoded.hidden_states.is_some());
    let (path, _handle) = ferritin_test_data::TestFile::amplify_output_01().create_temp()?;
    let example_data = candle_core::safetensors::load(path, &Device::Cpu)?;
    // println!("Example Data: {:?}", example_data);
    for (idx, attention) in encoded.attentions.unwrap().iter().enumerate() {
        println!("idx: {:?}, attention: {:?}", idx, attention);
        let ref_data = example_data
            .get(format!("attention_{:?}", idx).as_str())
            .ok_or(std::fmt::Error)?;

        let tolerance = 1e-5;
        let tensor01 = ref_data.flatten_all()?.to_vec1::<f32>()?;
        let tensor02 = ref_data.flatten_all()?.to_vec1::<f32>()?;

        for (i, (value, expected_value)) in tensor01.iter().zip(tensor02.iter()).enumerate() {
            let difference = (value - expected_value).abs();
            assert!(
                difference < tolerance,
                "Error at index {}: value = {}, expected = {}. Difference = {} exceeds tolerance = {}.",
                i,
                value,
                expected_value,
                difference,
                tolerance
            );
        }
    }

    Ok(())
}
