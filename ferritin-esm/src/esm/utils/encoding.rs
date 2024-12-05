use candle_core::{Device, Result, Shape, Tensor};

use crate::models::vqvae::StructureTokenEncoder;
use crate::tokenization::function_tokenizer::InterProQuantizedTokenizer as EsmFunctionTokenizer;
use crate::tokenization::residue_tokenizer::ResidueAnnotationsTokenizer;
use crate::tokenization::sasa_tokenizer::SASADiscretizingTokenizer;
use crate::tokenization::sequence_tokenizer::EsmSequenceTokenizer;
use crate::tokenization::ss_tokenizer::SecondaryStructureTokenizer;
use crate::tokenization::structure_tokenizer::StructureTokenizer;
use crate::utils::constants::esm3 as C;
use crate::utils::function::encode_decode::encode_function_annotations;
use crate::utils::structure::protein_chain::ProteinChain;
use crate::utils::types::FunctionAnnotation;

// Raw Defaults
pub fn get_default_sequence(sequence_length: usize) -> String {
    C::MASK_STR_SHORT.repeat(sequence_length)
}

pub fn get_default_secondary_structure(sequence_length: usize) -> String {
    C::MASK_STR_SHORT.repeat(sequence_length)
}

pub fn get_default_sasa(sequence_length: usize) -> Vec<Option<f32>> {
    vec![None; sequence_length]
}

// Tokenization
pub fn tokenize_sequence(
    sequence: &str,
    sequence_tokenizer: &EsmSequenceTokenizer,
    add_special_tokens: bool,
) -> Result<Tensor> {
    let sequence = sequence.replace(C::MASK_STR_SHORT, sequence_tokenizer.mask_token());
    let sequence_tokens = sequence_tokenizer.encode(&sequence, add_special_tokens)?;
    Tensor::from_vec(sequence_tokens, sequence.len(), &Device::Cpu)
}

pub fn tokenize_structure(
    coordinates: &Tensor,
    structure_encoder: &StructureTokenEncoder,
    structure_tokenizer: &StructureTokenizer,
    reference_sequence: Option<&str>,
    add_special_tokens: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = coordinates.device();
    let chain = ProteinChain::from_atom37(coordinates, reference_sequence)?;

    // Setup padding
    if let Some(reference_sequence) = reference_sequence {
        if reference_sequence.len() != coordinates.dims().first()? {
            return Err(candle_core::Error::Msg(
                format!("Reference sequence length ({}) does not match number of residues in coordinates ({})",
                reference_sequence.len(), coordinates.dims().first()?)
            ));
        }
    }

    let mut left_pad = 0;
    let mut right_pad = 0;

    if add_special_tokens {
        left_pad += 1; // Add space for BOS token
        right_pad += 1; // Add space for EOS token
    }

    let (coordinates, plddt, residue_index) = chain.to_structure_encoder_inputs()?;
    let coordinates = coordinates.to_device(device)?;
    let plddt = plddt.to_device(device)?;
    let residue_index = residue_index.to_device(device)?;

    let (_, structure_tokens) = structure_encoder.encode(&coordinates, Some(&residue_index))?;

    let coordinates = coordinates.squeeze(0)?;
    let plddt = plddt.squeeze(0)?;
    let mut structure_tokens = structure_tokens.squeeze(0)?;

    // Add space for BOS and EOS tokens
    if add_special_tokens {
        let coordinates = coordinates.pad([left_pad, right_pad, 0, 0, 0, 0], f32::INFINITY)?;
        let plddt = plddt.pad([left_pad, right_pad], 0.0)?;
        structure_tokens =
            structure_tokens.pad([left_pad, right_pad], structure_tokenizer.mask_token_id())?;

        structure_tokens.slice_assign(0..1, structure_tokenizer.bos_token_id())?;
        structure_tokens.slice_assign(
            structure_tokens.dims().first()? - 1..,
            structure_tokenizer.eos_token_id(),
        )?;
    }

    Ok((coordinates, plddt, structure_tokens))
}

pub fn tokenize_secondary_structure(
    secondary_structure: &str,
    secondary_structure_tokenizer: &SecondaryStructureTokenizer,
    add_special_tokens: bool,
) -> Result<Tensor> {
    let mut tokens = secondary_structure
        .chars()
        .map(|c| {
            if c == C::MASK_STR_SHORT.chars().next().unwrap() {
                secondary_structure_tokenizer.mask_token()
            } else {
                c.to_string()
            }
        })
        .collect::<Vec<_>>();

    let tokens = secondary_structure_tokenizer.encode(&tokens, add_special_tokens)?;
    Tensor::from_vec(tokens, tokens.len(), &Device::Cpu)
}

pub fn tokenize_sasa(
    sasa: &[Option<f32>],
    sasa_tokenizer: &SASADiscretizingTokenizer,
    add_special_tokens: bool,
) -> Result<Tensor> {
    let tokens = sasa
        .iter()
        .map(|v| v.map(|x| x).unwrap_or_else(|| sasa_tokenizer.mask_token()))
        .collect::<Vec<_>>();

    let tokens = sasa_tokenizer.encode(&tokens, add_special_tokens)?;
    Tensor::from_vec(tokens, tokens.len(), &Device::Cpu)
}

pub fn tokenize_function_annotations(
    function_annotations: &[FunctionAnnotation],
    reference_sequence: &str,
    function_tokenizer: &EsmFunctionTokenizer,
    residue_annotation_tokenizer: &ResidueAnnotationsTokenizer,
    add_special_tokens: bool,
) -> Result<(Tensor, Tensor)> {
    let (function_tokens, residue_annotation_tokens) = encode_function_annotations(
        reference_sequence,
        function_annotations,
        function_tokenizer,
        residue_annotation_tokenizer,
        add_special_tokens,
    )?;

    Ok((function_tokens, residue_annotation_tokens))
}

// Tokenized Defaults
pub fn get_default_sequence_tokens(
    sequence_length: usize,
    sequence_tokenizer: &EsmSequenceTokenizer,
) -> Result<Tensor> {
    let mut tokens = vec![sequence_tokenizer.mask_token_id(); sequence_length + 2];
    tokens[0] = sequence_tokenizer.bos_token_id();
    tokens[tokens.len() - 1] = sequence_tokenizer.eos_token_id();
    Tensor::from_vec(tokens, sequence_length + 2, &Device::Cpu)
}

pub fn get_default_structure_tokens(
    sequence_length: usize,
    structure_tokenizer: &StructureTokenizer,
) -> Result<Tensor> {
    let mut tokens = vec![structure_tokenizer.mask_token_id(); sequence_length + 2];
    tokens[0] = structure_tokenizer.bos_token_id();
    tokens[tokens.len() - 1] = structure_tokenizer.eos_token_id();
    Tensor::from_vec(tokens, sequence_length + 2, &Device::Cpu)
}

pub fn get_default_secondary_structure_tokens(
    sequence_length: usize,
    secondary_structure_tokenizer: &SecondaryStructureTokenizer,
) -> Result<Tensor> {
    let mut tokens = vec![secondary_structure_tokenizer.mask_token_id(); sequence_length + 2];
    tokens[0] = secondary_structure_tokenizer.bos_token_id();
    tokens[tokens.len() - 1] = secondary_structure_tokenizer.eos_token_id();
    Tensor::from_vec(tokens, sequence_length + 2, &Device::Cpu)
}

pub fn get_default_sasa_tokens(
    sequence_length: usize,
    sasa_tokenizer: &SASADiscretizingTokenizer,
) -> Result<Tensor> {
    let mut tokens = vec![sasa_tokenizer.mask_token_id(); sequence_length + 2];
    tokens[0] = sasa_tokenizer.bos_token_id();
    tokens[tokens.len() - 1] = sasa_tokenizer.eos_token_id();
    Tensor::from_vec(tokens, sequence_length + 2, &Device::Cpu)
}

pub fn get_default_function_tokens(
    sequence_length: usize,
    function_tokenizer: &EsmFunctionTokenizer,
) -> Result<Tensor> {
    let shape = [sequence_length + 2, function_tokenizer.depth()];
    let mut tokens = Tensor::ones(shape, &Device::Cpu)?;
    tokens.mul_scalar(function_tokenizer.pad_token_id() as f64)?;

    tokens.slice_assign(0..1, function_tokenizer.bos_token_id())?;
    tokens.slice_assign(sequence_length + 1.., function_tokenizer.eos_token_id())?;

    Ok(tokens)
}

pub fn get_default_residue_annotation_tokens(
    sequence_length: usize,
    residue_annotation_tokenizer: &ResidueAnnotationsTokenizer,
) -> Result<Tensor> {
    let shape = [sequence_length + 2, C::MAX_RESIDUE_ANNOTATIONS];
    let mut tokens = Tensor::ones(shape, &Device::Cpu)?;
    tokens.mul_scalar(residue_annotation_tokenizer.pad_token_id() as f64)?;

    tokens.slice_assign(0..1, residue_annotation_tokenizer.bos_token_id())?;
    tokens.slice_assign(
        sequence_length + 1..,
        residue_annotation_tokenizer.eos_token_id(),
    )?;

    Ok(tokens)
}
