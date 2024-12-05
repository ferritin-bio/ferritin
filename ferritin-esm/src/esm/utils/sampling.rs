use candle_core::{DType, Device, Result, Shape, Tensor};
use std::fmt;
use thiserror::Error;

const MAX_RESIDUE_ANNOTATIONS: usize = 32;
const SASA_DISCRETIZATION_BOUNDARIES: &[f32] = &[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9];

pub fn non_batched_dims(k: &str, tensor: &Tensor) -> Result<usize> {
    match k {
        "sequence" => Ok(1),
        "structure" => {
            if tensor.dtype() == DType::F32 {
                Ok(2)
            } else {
                Ok(1)
            }
        }
        "secondary_structure" => Ok(1),
        "sasa" => Ok(1),
        "function" => Ok(2),
        "residue_annotations" => Ok(2),
        "coordinates" => Ok(3),
        _ => Err(RotationError::InvalidTrackDim.into()),
    }
}

#[derive(Debug, Clone)]
pub struct BatchedESMProteinTensor {
    sequence: Option<Tensor>,
    structure: Option<Tensor>,
    secondary_structure: Option<Tensor>,
    sasa: Option<Tensor>,
    function: Option<Tensor>,
    residue_annotations: Option<Tensor>,
    coordinates: Option<Tensor>,
}

impl BatchedESMProteinTensor {
    pub fn from_protein_tensor(protein: ESMProteinTensor) -> Result<Self> {
        fn maybe_unsqueeze(tensor: Option<Tensor>) -> Result<Option<Tensor>> {
            match tensor {
                Some(t) => Ok(Some(t.unsqueeze(0)?)),
                None => Ok(None),
            }
        }

        Ok(Self {
            sequence: maybe_unsqueeze(protein.sequence)?,
            structure: maybe_unsqueeze(protein.structure)?,
            secondary_structure: maybe_unsqueeze(protein.secondary_structure)?,
            sasa: maybe_unsqueeze(protein.sasa)?,
            function: maybe_unsqueeze(protein.function)?,
            residue_annotations: maybe_unsqueeze(protein.residue_annotations)?,
            coordinates: maybe_unsqueeze(protein.coordinates)?,
        })
    }
}

pub fn get_default_sampling_config(
    tokenizers: &TokenizerCollectionProtocol,
) -> Result<SamplingConfig> {
    let mut config = SamplingConfig::default();
    for track in [
        "sequence",
        "structure",
        "secondary_structure",
        "sasa",
        "function",
    ] {
        let invalid_ids = tokenizers.get_invalid_ids(track)?;
        let only_mask = !["secondary_structure", "sasa", "function"].contains(&track);

        config.set_track_config(
            track,
            SamplingTrackConfig {
                invalid_ids,
                temperature: 1.0,
                top_p: 1.0,
                only_sample_masked_tokens: only_mask,
            },
        )?;
    }
    Ok(config)
}

pub fn validate_sampling_config(config: &SamplingConfig, on_invalid: &str) -> Result<()> {
    for (track, max_topk) in config.track_max_topks() {
        if let Some(track_config) = config.get_track_config(track) {
            if track_config.topk_logprobs > max_topk {
                let msg = format!(
                    "Sampling track {} has topk_logprobs={} greater than MAX_TOPK={}",
                    track, track_config.topk_logprobs, max_topk
                );
                match on_invalid {
                    "raise" => return Err(RotationError::InvalidTopK(msg).into()),
                    _ => println!("Warning: {}", msg),
                }
            }
        }
    }
    Ok(())
}

pub fn sample_logits(
    logits: &Tensor,
    temperature: f64,
    valid_ids: &[usize],
    top_p: f64,
    mask_invalid: bool,
) -> Result<Tensor> {
    if valid_ids.is_empty() {
        return Err(RotationError::NoValidIds.into());
    }

    let mut logits = if top_p < 1.0 {
        top_p_logits(logits, top_p)?
    } else {
        logits.clone()
    };

    let batch_dims = &logits.dims()[..logits.dims().len() - 1];
    logits = logits.reshape(([-1, logits.dims().last().unwrap()]))?;

    if mask_invalid {
        let mut mask = Tensor::ones_like(&logits)?.to_dtype(DType::Bool)?;
        let valid_ids = Tensor::from_slice(valid_ids, logits.device())?;
        mask.narrow(1, &valid_ids)?.fill_(false)?;
        logits = logits.where_cond(&mask, &Tensor::full_like(&logits, -f64::INFINITY)?)?;
    }

    let probs = logits.softmax(-1)?;
    let ids = probs.multinomial(1)?;

    Ok(ids.reshape(batch_dims)?)
}

pub fn sample_function_logits(
    logits: &Tensor,
    tokenizer: &InterProQuantizedTokenizer,
    top_p: f64,
    temperature: f64,
    p_none_threshold: f64,
) -> Result<(Tensor, Tensor)> {
    let [batch, seq_len, depth, vocab] = logits.dims4()?;
    assert_eq!(depth, tokenizer.depth());

    let logits = if top_p < 1.0 {
        top_p_logits(&logits, top_p)?
    } else {
        logits.clone()
    };

    let temperature = Tensor::ones(&[batch, seq_len, depth], logits.device())? * temperature;
    let log_p = (logits / temperature.unsqueeze(-1)?).log_softmax(-1)?;

    let none_idx = tokenizer.get_idx("<none>")?;
    let log_p_nones = log_p.get(none_idx)?;
    let p_none = log_p_nones.exp()?.mean(-1)?;
    let where_none = p_none.gt(p_none_threshold)?;

    let expanded_where_not_none = !where_none.unsqueeze(-1)?.unsqueeze(-1)?;
    let none_mask = Tensor::arange(0, vocab, logits.device())?.eq(none_idx)?;
    let mask = expanded_where_not_none.broadcast_as(&[batch, seq_len, depth, vocab])? & none_mask;

    let log_p = log_p.where_cond(&mask, &Tensor::full_like(&log_p, -f64::INFINITY)?)?;
    let mut ids = log_p.argmax(-1)?;
    ids = ids.where_cond(
        &where_none.unsqueeze(-1)?,
        &Tensor::full_like(&ids, none_idx)?,
    )?;

    Ok((ids, log_p))
}

pub fn sample_residue_annotation_logits(
    logits: &Tensor,
    annotation_threshold: f64,
) -> Result<(Tensor, Tensor)> {
    let top_k_values = logits.topk(MAX_RESIDUE_ANNOTATIONS, true)?;
    let top_residue_annotations_idx = top_k_values.indices;
    let top_residue_annotations_logprobs = top_k_values.values.sigmoid()?.log()?;

    let is_negative = top_residue_annotations_logprobs.lt(annotation_threshold)?;
    let top_residue_annotations_idx = top_residue_annotations_idx.where_cond(
        &is_negative,
        &Tensor::zeros_like(&top_residue_annotations_idx)?,
    )?;

    Ok((
        top_residue_annotations_idx,
        top_residue_annotations_logprobs,
    ))
}

pub fn sample_sasa_logits(
    logits: &Tensor,
    tokens: &Tensor,
    sampling_track_config: &SamplingTrackConfig,
    mask_idx: usize,
    valid_ids: &[usize],
    mask_logits_of_invalid_ids: bool,
) -> Result<Tensor> {
    let logits = if mask_logits_of_invalid_ids {
        let mut mask = Tensor::ones_like(logits)?.to_dtype(DType::Bool)?;
        let valid_ids = Tensor::from_slice(valid_ids, logits.device())?;
        mask.narrow(1, &valid_ids)?.fill_(false)?;
        logits.where_cond(&mask, &Tensor::full_like(&logits, -f64::INFINITY)?)?
    } else {
        logits.clone()
    };

    let sasa_probs = logits.softmax(-1)?;
    let max_prob_idx = sasa_probs.argmax(-1)?;

    let sasa_bins = Tensor::from_slice(&SASA_DISCRETIZATION_BOUNDARIES, logits.device())?;
    let sasa_bins = ((sasa_bins.slice(0, 0, -1)? + sasa_bins.slice(0, 1, None)?) / 2.0)?;

    let sampling_mask = get_sampling_mask(tokens, sampling_track_config, mask_idx)?;

    let sasa_value = (sasa_probs.slice(-1, 3, -1)? * sasa_bins).sum(-1)?;
    let sasa_value = sasa_value.where_cond(
        &max_prob_idx.eq(18)?,
        &Tensor::full_like(&sasa_value, f64::INFINITY)?,
    )?;
    let sasa_value = sasa_value.where_cond(
        &sampling_mask,
        &Tensor::full_like(&sasa_value, f64::INFINITY)?,
    )?;

    Ok(sasa_value)
}

pub fn top_p_logits(logits: &Tensor, top_p: f64) -> Result<Tensor> {
    let batch_dims = &logits.dims()[..logits.dims().len() - 1];
    let logits = logits.reshape(([-1, logits.dims().last().unwrap()]))?;

    let sorted = logits.sort_descending(-1)?;
    let cumsum = sorted.values.softmax(-1)?.cumsum(-1)?;
    let top_p_mask = cumsum.le(top_p)?;
    top_p_mask.narrow(1, 0, 1)?.fill_(true)?;

    let logits = logits.where_cond(&top_p_mask, &Tensor::full_like(&logits, f64::NEG_INFINITY)?)?;

    Ok(logits.reshape(batch_dims)?)
}

pub fn get_sampling_mask(
    tokens: &Tensor,
    sampling_track_config: &SamplingTrackConfig,
    mask_idx: usize,
) -> Result<Tensor> {
    let mut sampling_mask = Tensor::ones_like(tokens)?.to_dtype(DType::Bool)?;
    sampling_mask.narrow(1, 0, 1)?.fill_(false)?;
    sampling_mask.narrow(1, -1, 1)?.fill_(false)?;

    let special_tokens: Vec<usize> = sampling_track_config
        .invalid_ids
        .iter()
        .filter(|&&id| id != mask_idx)
        .copied()
        .collect();

    if !special_tokens.is_empty() {
        let special_tokens = Tensor::from_slice(&special_tokens, tokens.device())?;
        let token_mask = tokens.unsqueeze(-1)?.ne(&special_tokens)?.all(-1)?;
        sampling_mask = sampling_mask & token_mask;
    }

    if sampling_track_config.only_sample_masked_tokens {
        let masked_tokens = tokens.eq(mask_idx)?;
        sampling_mask = sampling_mask & masked_tokens;
    }

    Ok(sampling_mask)
}

#[derive(Error, Debug)]
pub enum RotationError {
    #[error("Invalid rotation matrix shape")]
    InvalidShape,
    #[error("Invalid track dimension")]
    InvalidTrackDim,
    #[error("No valid IDs provided for sampling")]
    NoValidIds,
    #[error("Invalid top-k value: {0}")]
    InvalidTopK(String),
}
