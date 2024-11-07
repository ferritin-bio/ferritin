use candle_core::{DType, Device, Result, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinTokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    pad_token_id: usize,
    mask_token_id: usize,
    bos_token_id: usize,
    eos_token_id: usize,
    unk_token_id: usize,
    special_token_ids: HashSet<usize>,
}

impl ProteinTokenizer {
    pub fn new(
        vocab_path: &Path,
        pad_token_id: usize,
        mask_token_id: usize,
        bos_token_id: usize,
        eos_token_id: usize,
        unk_token_id: usize,
        other_special_token_ids: Option<Vec<usize>>,
    ) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Read vocabulary file
        let file = File::open(vocab_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open vocab file: {}", e)))?;
        let reader = BufReader::new(file);

        for (i, line) in reader.lines().enumerate() {
            let token =
                line.map_err(|e| candle_core::Error::Msg(format!("Failed to read line: {}", e)))?;
            let token = token.trim().to_string();
            token_to_id.insert(token.clone(), i);
            id_to_token.insert(i, token);
        }

        // Create special tokens set
        let mut special_token_ids = HashSet::new();
        special_token_ids.insert(pad_token_id);
        special_token_ids.insert(mask_token_id);
        special_token_ids.insert(bos_token_id);
        special_token_ids.insert(eos_token_id);
        special_token_ids.insert(unk_token_id);

        if let Some(other_tokens) = other_special_token_ids {
            special_token_ids.extend(other_tokens);
        }

        Ok(Self {
            token_to_id,
            id_to_token,
            pad_token_id,
            mask_token_id,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            special_token_ids,
        })
    }

    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn token_to_id(&self, token: &str) -> usize {
        *self.token_to_id.get(token).unwrap_or(&self.unk_token_id)
    }

    pub fn id_to_token(&self, index: usize) -> String {
        self.id_to_token
            .get(&index)
            .map(String::from)
            .unwrap_or_else(|| self.id_to_token[&self.unk_token_id].clone())
    }

    pub fn encode(
        &self,
        tokens: &[String],
        max_length: Option<usize>,
        add_special_tokens: bool,
        random_truncate: bool,
    ) -> Result<Tensor> {
        let mut token_ids: Vec<usize> =
            tokens.iter().map(|token| self.token_to_id(token)).collect();

        if add_special_tokens {
            token_ids.insert(0, self.bos_token_id);
            token_ids.push(self.eos_token_id);
        }

        if let Some(max_len) = max_length {
            if max_len < token_ids.len() {
                let offset = if random_truncate {
                    let mut rng = rand::thread_rng();
                    rng.gen_range(0..token_ids.len() - max_len)
                } else {
                    0
                };
                token_ids = token_ids[offset..offset + max_len].to_vec();
            }
        }

        // Convert to Tensor
        Tensor::from_slice(&token_ids, (token_ids.len(),), &Device::Cpu)
    }

    pub fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String {
        let mut tokens: Vec<usize> = token_ids.to_vec();

        if skip_special_tokens {
            // Remove special tokens from start and end
            if !tokens.is_empty() && self.special_token_ids.contains(&tokens[0]) {
                tokens.remove(0);
            }
            if !tokens.is_empty() && self.special_token_ids.contains(tokens.last().unwrap()) {
                tokens.pop();
            }
        }

        tokens
            .iter()
            .map(|&id| self.id_to_token(id))
            .collect::<Vec<String>>()
            .join(" ")
    }

    // Helper method to decode from Tensor
    pub fn decode_tensor(&self, tensor: &Tensor, skip_special_tokens: bool) -> Result<String> {
        let token_ids: Vec<usize> = tensor.to_vec1()?.into_iter().map(|x| x as usize).collect();
        Ok(self.decode(&token_ids, skip_special_tokens))
    }
}

// Implementation of additional utility traits
impl Default for ProteinTokenizer {
    fn default() -> Self {
        Self::new(
            Path::new("vocab.txt"),
            0, // pad_token_id
            1, // mask_token_id
            2, // bos_token_id
            3, // eos_token_id
            4, // unk_token_id
            None,
        )
        .unwrap()
    }
}

// use std::path::Path;

// fn main() -> Result<()> {
//     let tokenizer = ProteinTokenizer::new(
//         Path::new("vocab.txt"),
//         0,  // pad_token_id
//         1,  // mask_token_id
//         2,  // bos_token_id
//         3,  // eos_token_id
//         4,  // unk_token_id
//         None,
//     )?;

//     // Encode example
//     let tokens = vec!["A".to_string(), "B".to_string(), "C".to_string()];
//     let encoded = tokenizer.encode(&tokens, Some(10), true, true)?;

//     // Decode example
//     let decoded = tokenizer.decode_tensor(&encoded, true)?;
//     println!("Decoded: {}", decoded);

//     Ok(())
// }
