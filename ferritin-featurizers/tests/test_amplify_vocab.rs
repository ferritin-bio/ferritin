use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use ferritin_featurizers::ProteinTokenizer;

#[test]
fn test_amplify_tokens() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "chandar-lab/AMPLIFY_120M";
    let revision = "main";
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let tokenizer = repo.get("tokenizer.json").unwrap();
    let protein_tokenizer = ProteinTokenizer::new(tokenizer).unwrap();

    let tokens = vec![
        "M".to_string(),
        "E".to_string(),
        "T".to_string(),
        "V".to_string(),
        "A".to_string(),
        "L".to_string(),
        "<pad>".to_string(),
        "<unk>".to_string(),
        "<mask>".to_string(),
        "<bos>".to_string(),
        "<eos>".to_string(),
    ];

    let encoded = protein_tokenizer.encode(&tokens, Some(20), true, true)?;
    let decoded = protein_tokenizer.decode(
        &encoded
            .to_vec1::<i64>()?
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<u32>>(),
        true,
    )?;

    // Check vocab size
    assert_eq!(protein_tokenizer.len(), 27);
    // Check roundtrip decoding
    assert_eq!(&decoded, "M E T V A L");

    Ok(())
}
