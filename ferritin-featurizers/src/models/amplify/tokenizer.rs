use anyhow::{anyhow, Result};
use candle_core::Tensor;
use rand;
use std::path::Path;
use tokenizers::Encoding;
// use tokenizers::Result as TokenResult;
use tokenizers::Tokenizer;

pub struct ProteinTokenizer {
    tokenizer: Tokenizer,
    pad_token_id: u32,
    mask_token_id: u32,
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
    special_token_ids: std::collections::HashSet<u32>,
}

impl ProteinTokenizer {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        // Load the tokenizer from file
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Get special token IDs
        let pad_token_id = tokenizer
            .token_to_id("<pad>")
            .ok_or_else(|| anyhow!("Missing pad token"))? as u32;
        let mask_token_id = tokenizer
            .token_to_id("<mask>")
            .ok_or_else(|| anyhow!("Missing mask token"))? as u32;
        let bos_token_id = tokenizer
            .token_to_id("<bos>")
            .ok_or_else(|| anyhow!("Missing bos token"))? as u32;
        let eos_token_id = tokenizer
            .token_to_id("<eos>")
            .ok_or_else(|| anyhow!("Missing eos token"))? as u32;
        let unk_token_id = tokenizer
            .token_to_id("<unk>")
            .ok_or_else(|| anyhow!("Missing unk token"))? as u32;

        // Create set of special token IDs
        let mut special_token_ids = std::collections::HashSet::new();
        special_token_ids.insert(pad_token_id);
        special_token_ids.insert(mask_token_id);
        special_token_ids.insert(bos_token_id);
        special_token_ids.insert(eos_token_id);
        special_token_ids.insert(unk_token_id);

        Ok(Self {
            tokenizer,
            pad_token_id,
            mask_token_id,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            special_token_ids,
        })
    }

    pub fn len(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> u32 {
        self.tokenizer
            .token_to_id(token)
            .unwrap_or(self.unk_token_id)
    }

    pub fn id_to_token(&self, id: u32) -> String {
        self.tokenizer
            .id_to_token(id)
            .unwrap_or_else(|| "<unk>".to_string())
    }

    pub fn encode(
        &self,
        tokens: &[String],
        max_length: Option<usize>,
        add_special_tokens: bool,
        random_truncate: bool,
    ) -> Result<Tensor> {
        // Join tokens with spaces as the tokenizer expects text
        let text = tokens.join(" ");

        let encoding: Encoding = if random_truncate && max_length.is_some() {
            let max_len = max_length.unwrap();
            if tokens.len() + 2 > max_len {
                let available_start = tokens.len() - max_len + 2;
                let offset = rand::random::<usize>() % available_start;
                let truncated_tokens = &tokens[offset..offset + max_len - 2];
                self.tokenizer
                    .encode(truncated_tokens.join(" ").as_str(), add_special_tokens)
                    .map_err(|e| anyhow!("Failed to encode truncated tokens: {}", e))?
            } else {
                self.tokenizer
                    .encode(text.as_str(), add_special_tokens)
                    .map_err(|e| anyhow!("Failed to encode text: {}", e))?
            }
        } else {
            self.tokenizer
                .encode(text.as_str(), add_special_tokens)
                .map_err(|e| anyhow!("Failed to encode text: {}", e))?
        };

        // Convert to Tensor
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();

        Tensor::new(ids, &candle_core::Device::Cpu)
            .map_err(|e| anyhow!("Failed to create tensor: {}", e))
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if skip_special_tokens {
            let filtered: Vec<u32> = token_ids
                .iter()
                .filter(|&&id| !self.special_token_ids.contains(&id))
                .copied()
                .collect();

            self.tokenizer
                .decode(&filtered, true)
                .map_err(|e| anyhow!("Failed to decode: {}", e))
        } else {
            self.tokenizer
                .decode(token_ids, true)
                .map_err(|e| anyhow!("Failed to decode: {}", e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_tokenizer() -> Result<()> {
    //     let tokenizer = ProteinTokenizer::new("path/to/tokenizer.json")?;

    //     let tokens = vec![
    //         "M".to_string(),
    //         "E".to_string(),
    //         "T".to_string(),
    //         "V".to_string(),
    //         "A".to_string(),
    //         "L".to_string(),
    //     ];

    //     let encoded = tokenizer.encode(&tokens, Some(10), true, true)?;
    //     let decoded = tokenizer.decode(&encoded.to_vec1::<u32>()?, true)?;

    //     println!("Encoded: {:?}", encoded);
    //     println!("Decoded: {}", decoded);

    //     Ok(())
    // }
}
