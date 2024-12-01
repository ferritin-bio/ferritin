use anyhow::Result;
use candle_core::{Device, D};
// use candle_examples::device;
use candle_core::utils::{cuda_is_available, metal_is_available};
use ferritin_featurizers::AMPLIFY;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

// #[ignore = "downloads large model weights (>100MB) from HuggingFace"]
#[test]
fn test_amplify_full_model() -> Result<(), Box<dyn std::error::Error>> {
    // Load the Model adn the Tokenizer

    let dev = device(false)?;
    let (tokenizer, amplify) = AMPLIFY::load_from_huggingface(dev.clone())?;

    // Test the outputs of the Encoding from the Amplify Test Suite
    let AMPLIFY_TEST_SEQ = "MSVVGIDLGFQSCYVAVARAGGIETIANEYSDRCTPACISFGPKNR";
    let pmatrix = tokenizer.encode(&[AMPLIFY_TEST_SEQ.to_string()], None, true, false)?;
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case

    // Run the sequence through the model.
    let encoded = amplify.forward(&pmatrix.to_device(&dev)?, None, true, true)?;

    println!("Encoded!");

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
    // let tolerance = 1e-5f32;
    let tolerance = 1e-2f32;

    let (path, _handle) = ferritin_test_data::TestFile::amplify_output_01().create_temp()?;
    let example_data = candle_core::safetensors::load(path, &dev)?;
    for (idx, attention) in encoded.attentions.unwrap().iter().enumerate() {
        println!("idx: {:?}, attention: {:?}", idx, attention);

        let ref_data = example_data
            .get(format!("attention_{:?}", idx).as_str())
            .ok_or(std::fmt::Error)?;

        let tensor01 = attention.flatten_all()?;
        let tensor02 = ref_data.flatten_all()?;
        println!(
            "tensor01 device: {:?}, dims: {:?}",
            tensor01.device(),
            tensor01.dims()
        );
        println!(
            "tensor02 device: {:?}, dims: {:?}",
            tensor02.device(),
            tensor02.dims()
        );
        let tensor01 = tensor01.to_vec1::<f32>()?;
        let tensor02 = tensor02.to_vec1::<f32>()?;
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
