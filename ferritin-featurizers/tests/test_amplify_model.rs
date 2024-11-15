use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use ferritin_featurizers::{AMPLIFYConfig, ProteinTokenizer, AMPLIFY};

#[test]
#[ignore]
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

    // Example Data is the Saved Data
    let example_data = candle_core::safetensors::load(path, &Device::Cpu)?;
    let tolerance = 1e-5;
    for (idx, attention) in encoded.attentions.unwrap().iter().enumerate() {
        println!("idx: {:?}, attention: {:?}", idx, attention);
        let ref_data = example_data
            .get(format!("attention_{:?}", idx).as_str())
            .ok_or(std::fmt::Error)?;
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

// #[ignore]
#[test]
fn test_amplify_attentions() -> Result<(), Box<dyn std::error::Error>> {
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
    let sprot_01 = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";
    let pmatrix = protein_tokenizer.encode(&[sprot_01.to_string()], None, true, false)?;
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case
    let encoded = model.forward(&pmatrix, None, true, true)?;
    let predictions = &encoded.logits.argmax(D::Minus1)?;
    let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
    let decoded = protein_tokenizer.decode(indices.as_slice(), true)?;

    fn trim_ends(s: &str) -> &str {
        if s.len() >= 2 {
            &s[1..s.len() - 1]
        } else {
            ""
        }
    }

    assert_eq!(sprot_01, trim_ends(decoded.replace(" ", "").as_str()));
    assert!(&encoded.attentions.is_some());
    assert!(&encoded.hidden_states.is_some());

    let attention_map = &encoded.attentions.unwrap();
    println!("Attentions: {:?}", attention_map);
    assert_eq!(&attention_map.len(), &24);

    let attn_map_combined = Tensor::stack(&attention_map, 0)?;
    println!(
        "Attentions Combined: {:?}, {:?}",
        attn_map_combined.dims(),
        attn_map_combined
    );

    let last_dim = pmatrix.dim(D::Minus1)?;
    let total_elements = attn_map_combined.dims().iter().product::<usize>();
    let first_dim = total_elements / (last_dim * last_dim);
    let attn_map_combined2 = attn_map_combined.reshape((first_dim, last_dim, last_dim))?;

    // In PyTorch: attn_map = attn_map[:, 1:-1, 1:-1]
    let attn_map_combined2 = attn_map_combined2
        .narrow(1, 1, attn_map_combined2.dim(1)? - 2)? // second dim
        .narrow(2, 1, attn_map_combined2.dim(2)? - 2)?; // third dim

    println!(
        "Attentions Combined Reshaped: {:?}, {:?}",
        attn_map_combined.dims(),
        attn_map_combined2.dims()
    );

    /// "Perform average product correct, used for contact prediction."
    // https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/examples/utils.py#L83
    fn apc(x: &Tensor) -> Result<Tensor> {
        // "Perform average product correct, used for contact prediction."
        // Sum along last dimension (keeping dims)
        let a1 = x.sum_keepdim(D::Minus1)?;
        // Sum along second-to-last dimension (keeping dims)
        let a2 = x.sum_keepdim(D::Minus2)?;
        // Sum along both last dimensions (keeping dims)
        let a12 = x.sum_keepdim((D::Minus1, D::Minus2))?;
        // Multiply a1 and a2
        let avg = a1.matmul(&a2)?;
        // Divide by a12 (equivalent to pytorch's div_)
        // println!("IN the APC: avg, a12 {:?}, {:?}", avg, a12);
        // let avg = avg.div(&a12)?;
        let a12_broadcast = a12.broadcast_as(avg.shape())?;
        // Divide by a12 (with proper broadcasting)
        let avg = avg.div(&a12_broadcast)?;
        // Subtract avg from x
        Ok(x.sub(&avg)?)
    }

    //From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
    // https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/examples/utils.py#L77
    fn symmetrize(x: &Tensor) -> Result<Tensor> {
        // "Make layer symmetric in final two dimensions, used for contact prediction."
        let x_transpose = x.transpose(D::Minus1, D::Minus2)?;
        Ok(x.add(&x_transpose)?)
    }

    // attn_map = apc(symmetrize(attn_map))  # process the attention maps
    // attn_map = attn_map.permute(1, 2, 0)  # (residues, residues, map)
    let symmetric = symmetrize(&attn_map_combined2)?;
    let normalized = apc(&symmetric)?;
    let proximity_map = normalized.permute((1, 2, 0)); //  # (residues, residues, map)

    println!(
        "sym, norm, proxmap: {:?}, {:?}, {:?}",
        symmetric, normalized, proximity_map
    );

    Ok(())
}
