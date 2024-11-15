use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use ferritin_featurizers::{AMPLIFYConfig, ProteinTokenizer, AMPLIFY};

#[test]
// #[ignore = "downloads large model weights (>100MB) from HuggingFace"]
fn test_amplify_full_model() -> Result<(), Box<dyn std::error::Error>> {
    // Load the Model adn the Tokenizer
    //
    let (tokenizer, amplify) = AMPLIFY::load_from_huggingface()?;

    // Test the outputs of the Encoding from the Amplify Test Suite
    //
    let AMPLIFY_TEST_SEQ = "MSVVGIDLGFQSCYVAVARAGGIETIANEYSDRCTPACISFGPKNR";
    let pmatrix = tokenizer.encode(&[AMPLIFY_TEST_SEQ.to_string()], None, true, false)?;
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case

    // Run the sequence through the model.
    let encoded = amplify.forward(&pmatrix, None, true, true)?;

    // Choosing ARGMAX. We expect this to be the most predicted sequence.
    // it should return the identity of an unmasked sequence
    let predictions = &encoded.logits.argmax(D::Minus1)?;
    let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
    let decoded = tokenizer.decode(indices.as_slice(), true)?;
    let final_seq = format!("S{}S", AMPLIFY_TEST_SEQ);

    assert_eq!(final_seq, decoded.replace(" ", ""));
    assert!(&encoded.attentions.is_some());
    assert!(&encoded.hidden_states.is_some());

    // Now test the outputs of the attention vectors.
    // for each member of the output, assert that the difference
    // between the saved pytorch model and the Candle model is
    // less than a tolerance.
    //
    let tolerance = 1e-5f32;
    let (path, _handle) = ferritin_test_data::TestFile::amplify_output_01().create_temp()?;
    let example_data = candle_core::safetensors::load(path, &Device::Cpu)?;
    for (idx, attention) in encoded.attentions.unwrap().iter().enumerate() {
        let ref_data = example_data
            .get(format!("attention_{:?}", idx).as_str())
            .ok_or(std::fmt::Error)?;

        let max_diff = attention
            .sub(ref_data)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;

        assert!(
            max_diff < tolerance,
            "Error between pytorch test set and Candle set exceeds tolerance. Tolerance: {}, Actual difference: {}, at attention layer {}",
                    tolerance,
                    max_diff,
                    idx
        );
    }

    // Check that we can retrieve the contact map from the Model Outputs.
    //

    // Run a bigger sequence through the model.
    let sprot_01 = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";
    let pmatrix = tokenizer.encode(&[sprot_01.to_string()], None, true, false)?;
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case
    let encoded_long = amplify.forward(&pmatrix, None, true, true)?;

    if let Some(norm) = &encoded_long.get_contact_map()? {
        assert_eq!(norm.dims3()?, (256, 256, 240));
    }

    // Check that we can run Multiple sequences through the model:
    //
    // todo!();

    Ok(())
}
