use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::models::bpe::BPE;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::Tokenizer;

pub trait EsmTokenizerBase {
    fn encode(&self) -> Result<()>;
    fn decode(&self) -> Result<()>;
    fn mask_token(&self) -> &str;
    fn mask_token_id(&self) -> u32;
    fn bos_token(&self) -> &str;
    fn bos_token_id(&self) -> u32;
    fn eos_token(&self) -> &str;
    fn eos_token_id(&self) -> u32;
    fn pad_token(&self) -> &str;
    fn pad_token_id(&self) -> u32;
    fn chain_break_token(&self) -> &str;
    fn chain_break_token_id(&self) -> u32;
    fn all_token_ids(&self) -> Vec<u32>;
    fn special_token_ids(&self) -> Vec<u32>;
}
pub struct EsmSequenceTokenizer {
    tokenizer: Arc<Tokenizer>,
    cb_token: String,
}

impl EsmSequenceTokenizer {
    pub fn new(
        unk_token: &str,
        cls_token: &str,
        pad_token: &str,
        mask_token: &str,
        eos_token: &str,
        chain_break_token: &str,
    ) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        for (i, tok) in SEQUENCE_VOCAB.iter().enumerate() {
            token_to_id.insert(tok.to_string(), i);
        }

        let bpe = BPE::new(token_to_id, vec![], unk_token.to_string());
        let mut tokenizer = Tokenizer::new(bpe);

        let special_tokens = vec![
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            chain_break_token,
        ];

        tokenizer.add_special_tokens(&special_tokens)?;

        let post_processor = TemplateProcessing::new(
            format!("{} $A {}", cls_token, eos_token),
            vec![
                (cls_token.to_string(), tokenizer.token_to_id(cls_token)?),
                (eos_token.to_string(), tokenizer.token_to_id(eos_token)?),
            ],
        )?;

        tokenizer.set_post_processor(Arc::new(post_processor));

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            cb_token: chain_break_token.to_string(),
        })
    }
}

impl EsmTokenizerBase for EsmSequenceTokenizer {
    fn encode(&self) -> Result<()> {
        todo!()
    }

    fn decode(&self) -> Result<()> {
        todo!()
    }

    fn mask_token(&self) -> &str {
        "mask"
    }

    fn mask_token_id(&self) -> u32 {
        self.tokenizer.token_to_id("mask").unwrap_or(0)
    }

    fn bos_token(&self) -> &str {
        self.cls_token()
    }

    fn bos_token_id(&self) -> u32 {
        self.cls_token_id()
    }

    fn eos_token(&self) -> &str {
        "eos"
    }

    fn eos_token_id(&self) -> u32 {
        self.tokenizer.token_to_id("eos").unwrap_or(0)
    }

    fn pad_token(&self) -> &str {
        "pad"
    }

    fn pad_token_id(&self) -> u32 {
        self.tokenizer.token_to_id("pad").unwrap_or(0)
    }

    fn chain_break_token(&self) -> &str {
        &self.cb_token
    }

    fn chain_break_token_id(&self) -> u32 {
        self.tokenizer.token_to_id(&self.cb_token).unwrap_or(0)
    }

    fn all_token_ids(&self) -> Vec<u32> {
        (0..self.vocab_size()).collect()
    }

    fn special_token_ids(&self) -> Vec<u32> {
        self.tokenizer.get_special_token_ids()
    }
}
