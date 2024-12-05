pub mod sequence_tokenizer;

use crate::esm::utils::constants::models::{normalize_model_name, ESM3_OPEN_SMALL};
use crate::tokenizer_base::EsmTokenizerBase;

pub struct TokenizerCollection {
    pub sequence: sequence_tokenizer::EsmSequenceTokenizer,
    pub structure: structure_tokenizer::StructureTokenizer,
    pub secondary_structure: ss_tokenizer::SecondaryStructureTokenizer,
    pub sasa: sasa_tokenizer::SASADiscretizingTokenizer,
    pub function: function_tokenizer::InterProQuantizedTokenizer,
    pub residue_annotations: residue_tokenizer::ResidueAnnotationsTokenizer,
}

pub fn get_model_tokenizers(model: &str) -> Result<TokenizerCollection> {
    if normalize_model_name(model) == ESM3_OPEN_SMALL {
        Ok(TokenizerCollection {
            sequence: sequence_tokenizer::EsmSequenceTokenizer::new()?,
            structure: structure_tokenizer::StructureTokenizer::new()?,
            secondary_structure: ss_tokenizer::SecondaryStructureTokenizer::new("ss8")?,
            sasa: sasa_tokenizer::SASADiscretizingTokenizer::new()?,
            function: function_tokenizer::InterProQuantizedTokenizer::new()?,
            residue_annotations: residue_tokenizer::ResidueAnnotationsTokenizer::new()?,
        })
    } else {
        Err(anyhow!("Unknown model: {}", model))
    }
}

pub fn get_invalid_tokenizer_ids(tokenizer: &impl EsmTokenizerBase) -> Vec<i64> {
    if tokenizer.is_sequence_tokenizer() {
        vec![
            tokenizer.mask_token_id(),
            tokenizer.pad_token_id(),
            tokenizer.cls_token_id(),
            tokenizer.eos_token_id(),
        ]
    } else {
        vec![
            tokenizer.mask_token_id(),
            tokenizer.pad_token_id(),
            tokenizer.bos_token_id(),
            tokenizer.eos_token_id(),
        ]
    }
}
