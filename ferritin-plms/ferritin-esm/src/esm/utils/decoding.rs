use candle_core::{DType, Tensor};
use candle_examples::device;
use std::warnings;

use crate::models::function_decoder::FunctionTokenDecoder;
use crate::models::vqvae::StructureTokenDecoder;
use crate::sdk::api::{ESMProtein, ESMProteinTensor};
use crate::tokenization::function_tokenizer::InterProQuantizedTokenizer;
use crate::tokenization::residue_tokenizer::ResidueAnnotationsTokenizer;
use crate::tokenization::sasa_tokenizer::SASADiscretizingTokenizer;
use crate::tokenization::sequence_tokenizer::EsmSequenceTokenizer;
use crate::tokenization::ss_tokenizer::SecondaryStructureTokenizer;
use crate::tokenization::structure_tokenizer::StructureTokenizer;
use crate::tokenization::tokenizer_base::EsmTokenizerBase;
use crate::tokenization::TokenizerCollectionProtocol;
use crate::utils::constants::esm3 as C;
use crate::utils::function::encode_decode::{
    decode_function_tokens, decode_residue_annotation_tokens,
};
use crate::utils::misc::maybe_list;
use crate::utils::structure::protein_chain::ProteinChain;
use crate::utils::types::FunctionAnnotation;

pub fn decode_protein_tensor(
    input: ESMProteinTensor,
    tokenizers: &TokenizerCollectionProtocol,
    structure_token_decoder: &StructureTokenDecoder,
    function_token_decoder: Option<&FunctionTokenDecoder>,
) -> ESMProtein {
    let mut input = input.clone();

    let mut sequence = None;
    let mut secondary_structure = None;
    let mut sasa = None;
    let mut function_annotations = Vec::new();

    let mut coordinates = None;

    // If all pad tokens, set to None
    for field in ESMProteinTensor::fields() {
        let tokens = input.get_field(&field.name);
        if field.name == "coordinates" || field.name == "potential_sequence_of_concern" {
            continue;
        }
        if let Some(tokens) = tokens {
            let tokens = &tokens.slice(1..-1, 0); // Remove BOS and EOS tokens
            let tokens = tokens.reshape(&[-1])?;
            let track_tokenizer = tokenizers.get_tokenizer(&field.name);
            if tokens.eq(track_tokenizer.pad_token_id)? {
                input.set_field(&field.name, None);
            }
            // If structure track has any mask tokens, do not decode
            if field.name == "structure" && tokens.eq(track_tokenizer.mask_token_id)? {
                input.set_field(&field.name, None);
            }
        }
    }

    if let Some(seq_tokens) = input.sequence {
        sequence = Some(decode_sequence(&seq_tokens, &tokenizers.sequence));
    }

    let (plddt, ptm) = (None, None);
    if let Some(struct_tokens) = input.structure {
        let (coords, p, t) = decode_structure(
            &struct_tokens,
            structure_token_decoder,
            &tokenizers.structure,
            sequence.as_deref(),
        );
        coordinates = Some(coords);
        plddt = p;
        ptm = t;
    } else if let Some(coords) = input.coordinates {
        coordinates = Some(coords.slice(1..-1, 0..)?);
    }

    if let Some(ss_tokens) = input.secondary_structure {
        secondary_structure = Some(decode_secondary_structure(
            &ss_tokens,
            &tokenizers.secondary_structure,
        ));
    }

    if let Some(sasa_tokens) = input.sasa {
        sasa = Some(decode_sasa(&sasa_tokens, &tokenizers.sasa));
    }

    if let Some(fn_tokens) = input.function {
        let decoder = function_token_decoder
            .expect("Cannot decode function annotations without a function token decoder");
        let annotations = decode_function_annotations(&fn_tokens, decoder, &tokenizers.function);
        function_annotations.extend(annotations);
    }

    if let Some(res_tokens) = input.residue_annotations {
        let annotations = decode_residue_annotations(&res_tokens, &tokenizers.residue_annotations);
        function_annotations.extend(annotations);
    }

    ESMProtein::new(
        sequence,
        secondary_structure,
        sasa,
        if function_annotations.is_empty() {
            None
        } else {
            Some(function_annotations)
        },
        coordinates,
        plddt,
        ptm,
        input.potential_sequence_of_concern,
    )
}

fn bos_eos_warn(msg: &str, tensor: &Tensor, tok: &EsmTokenizerBase) {
    if tensor.get(0)? != tok.bos_token_id {
        warnings::warn!(
            "{} does not start with BOS token, token is ignored. BOS={} vs {}",
            msg,
            tok.bos_token_id,
            tensor
        );
    }
    if tensor.get(&[-1])? != tok.eos_token_id {
        warnings::warn!(
            "{} does not end with EOS token, token is ignored. EOS='{}': {}",
            msg,
            tok.eos_token_id,
            tensor
        );
    }
}

pub fn decode_sequence(
    sequence_tokens: &Tensor,
    sequence_tokenizer: &EsmSequenceTokenizer,
) -> String {
    bos_eos_warn("Sequence", sequence_tokens, sequence_tokenizer);
    let mut sequence = sequence_tokenizer.decode(sequence_tokens);
    sequence = sequence.replace(" ", "");
    sequence = sequence.replace(&sequence_tokenizer.mask_token, &C::MASK_STR_SHORT);
    sequence = sequence.replace(&sequence_tokenizer.cls_token, "");
    sequence = sequence.replace(&sequence_tokenizer.eos_token, "");
    sequence
}

fn decode_structure(
    structure_tokens: &Tensor,
    structure_decoder: &StructureTokenDecoder,
    structure_tokenizer: &StructureTokenizer,
    sequence: Option<&str>,
) -> (Tensor, Option<Tensor>, Option<Tensor>) {
    let is_singleton = structure_tokens.dims().len() == 1;
    let structure_tokens = if is_singleton {
        structure_tokens.unsqueeze(0)?
    } else {
        panic!(
            "Only one structure can be decoded at a time, got structure tokens of shape {:?}",
            structure_tokens.dims()
        );
    };

    bos_eos_warn("Structure", &structure_tokens.get(0)?, structure_tokenizer);

    let decoder_output = structure_decoder.decode(&structure_tokens);
    let bb_coords = decoder_output.bb_pred.get(0)?.slice(1..-1, 0..)?.cpu()?;

    let plddt = decoder_output
        .plddt
        .map(|p| p.get(0)?.slice(1..-1, 0..)?.cpu());

    let ptm = decoder_output.ptm;

    let chain = ProteinChain::from_backbone_atom_coordinates(&bb_coords, sequence);
    let chain = chain.infer_oxygen();

    (
        Tensor::new(chain.atom37_positions, structure_tokens.device())?,
        plddt,
        ptm,
    )
}

fn decode_secondary_structure(
    secondary_structure_tokens: &Tensor,
    ss_tokenizer: &SecondaryStructureTokenizer,
) -> String {
    bos_eos_warn(
        "Secondary structure",
        secondary_structure_tokens,
        ss_tokenizer,
    );
    let tokens = secondary_structure_tokens.slice(1..-1, 0..)?;
    ss_tokenizer.decode(&tokens)
}

fn decode_sasa(sasa_tokens: &Tensor, sasa_tokenizer: &SASADiscretizingTokenizer) -> Vec<f32> {
    if sasa_tokens.get(0)? != 0.0 {
        panic!("SASA does not start with 0 corresponding to BOS token");
    }
    if sasa_tokens.get(&[-1])? != 0.0 {
        panic!("SASA does not end with 0 corresponding to EOS token");
    }

    let tokens = sasa_tokens.slice(1..-1, 0..)?;

    match tokens.dtype() {
        DType::I64 | DType::I32 | DType::I16 | DType::I8 => sasa_tokenizer.decode_float(&tokens),
        _ => maybe_list(&tokens, true)
            .into_iter()
            .map(|x| x.unwrap_or(std::f32::NAN))
            .collect(),
    }
}

fn decode_function_annotations(
    function_annotation_tokens: &Tensor,
    function_token_decoder: &FunctionTokenDecoder,
    function_tokenizer: &InterProQuantizedTokenizer,
) -> Vec<FunctionAnnotation> {
    decode_function_tokens(
        function_annotation_tokens,
        function_token_decoder,
        function_tokenizer,
    )
}

fn decode_residue_annotations(
    residue_annotation_tokens: &Tensor,
    residue_annotation_decoder: &ResidueAnnotationsTokenizer,
) -> Vec<FunctionAnnotation> {
    decode_residue_annotation_tokens(residue_annotation_tokens, residue_annotation_decoder)
}
