use crate::esmc::utils::constants::esm3::SEQUENCE_VOCAB;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::models::bpe::{BPE, BpeBuilder};
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::processors::template::{Template, TemplateProcessing};
use tokenizers::{AddedToken, Tokenizer};

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
        let bpe_builder = BpeBuilder::new();
        let bpe: BPE = bpe_builder
            .unk_token(unk_token.to_string())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build BPE tokenizer: {}", e))?;

        let mut tokenizer = Tokenizer::new(bpe);
        let special_tokens = vec![
            AddedToken::from(cls_token, true),
            AddedToken::from(pad_token, true),
            AddedToken::from(mask_token, true),
            AddedToken::from(eos_token, true),
            AddedToken::from(chain_break_token, true),
        ];

        let _ = tokenizer.add_special_tokens(special_tokens);

        let post_processor = TemplateProcessing::builder()
            .try_single(Template::try_from(format!("{} $A {}", cls_token, eos_token)).unwrap())?
            .special_tokens(vec![
                (cls_token, tokenizer.token_to_id(cls_token).unwrap()),
                (eos_token, tokenizer.token_to_id(eos_token).unwrap()),
            ])
            .build()?;

        tokenizer.with_post_processor(Some(PostProcessorWrapper::Template(post_processor)));

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            cb_token: chain_break_token.to_string(),
        })
    }
}
impl Default for EsmSequenceTokenizer {
    fn default() -> Self {
        Self::new("<unk>", "<cls>", "<pad>", "<mask>", "<eos>", "|")
            .expect("Failed to create default tokenizer")
    }
}

impl EsmSequenceTokenizer {
    /// Tokenize an amino-acid sequence string into token IDs.
    ///
    /// Looks up each character in `SEQUENCE_VOCAB`. Unknown characters map to
    /// the `<unk>` token (index 3). When `add_special_tokens` is true, prepends
    /// BOS (`<cls>` = 0) and appends EOS (`<eos>` = 2).
    pub fn tokenize_sequence(&self, sequence: &str, add_special_tokens: bool) -> Vec<u32> {
        use std::collections::HashMap;
        let vocab: HashMap<&str, u32> = SEQUENCE_VOCAB
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u32))
            .collect();
        let unk_id = *vocab.get("<unk>").unwrap_or(&3);

        let mut tokens = Vec::with_capacity(sequence.len() + 2);
        if add_special_tokens {
            tokens.push(*vocab.get("<cls>").unwrap_or(&0));
        }
        for ch in sequence.chars() {
            let s = ch.to_string();
            let id = vocab.get(s.as_str()).copied().unwrap_or(unk_id);
            tokens.push(id);
        }
        if add_special_tokens {
            tokens.push(*vocab.get("<eos>").unwrap_or(&2));
        }
        tokens
    }

    /// Decode token IDs back to an amino-acid sequence string.
    ///
    /// Skips the standard special tokens (BOS=0, PAD=1, EOS=2, MASK=32) and
    /// concatenates the remaining vocabulary entries.
    pub fn decode_sequence(&self, token_ids: &[u32]) -> String {
        const SPECIAL: [u32; 4] = [0, 1, 2, 32]; // cls, pad, eos, mask
        let mut result = String::new();
        for &id in token_ids {
            if SPECIAL.contains(&id) {
                continue;
            }
            if let Some(tok) = SEQUENCE_VOCAB.get(id as usize) {
                result.push_str(tok);
            }
        }
        result
    }
}

impl EsmTokenizerBase for EsmSequenceTokenizer {
    fn encode(&self) -> Result<()> {
        // TODO: implement generic encode via the HuggingFace tokenizers Tokenizer
        todo!()
    }

    fn decode(&self) -> Result<()> {
        // TODO: implement generic decode via the HuggingFace tokenizers Tokenizer
        todo!()
    }

    fn mask_token(&self) -> &str {
        "mask"
    }

    fn mask_token_id(&self) -> u32 {
        self.tokenizer.token_to_id("mask").unwrap_or(0)
    }

    fn bos_token(&self) -> &str {
        // TODO: BOS is "<cls>" token — alias cls_token() once that method exists
        unimplemented!()
    }

    fn bos_token_id(&self) -> u32 {
        // TODO: BOS id is the "<cls>" token id — alias cls_token_id() once that method exists
        unimplemented!()
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
        // TODO: return (0..SEQUENCE_VOCAB.len() as u32).collect()
        unimplemented!()
    }

    fn special_token_ids(&self) -> Vec<u32> {
        // TODO: return ids for cls, pad, eos, mask, chain_break special tokens
        unimplemented!()
    }
}
