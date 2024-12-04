use abc::{abstractmethod, ABC};
use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;

use esm::tokenization::{get_model_tokenizers, TokenizerCollectionProtocol};
use esm::utils::constants::api as C;
use esm::utils::constants::models::ESM3_OPEN_SMALL;
use esm::utils::encoding;
use esm::utils::misc::get_chainbreak_boundaries_from_sequence;
use esm::utils::structure::protein_chain::ProteinChain;
use esm::utils::structure::protein_complex::ProteinComplex;
use esm::utils::types::{FunctionAnnotation, PathOrBuffer};

pub trait ProteinType {}

#[derive(Debug, Default)]
pub struct ESMProtein {
    sequence: Option<String>,
    secondary_structure: Option<String>,
    sasa: Option<Vec<Option<f32>>>,
    function_annotations: Option<Vec<FunctionAnnotation>>,
    coordinates: Option<Tensor>,
    plddt: Option<Tensor>,
    ptm: Option<Tensor>,
    potential_sequence_of_concern: bool,
}

impl ESMProtein {
    pub fn len(&self) -> usize {
        if let Some(seq) = &self.sequence {
            seq.len()
        } else if let Some(ss) = &self.secondary_structure {
            ss.len()
        } else if let Some(sasa) = &self.sasa {
            sasa.len()
        } else if let Some(coords) = &self.coordinates {
            coords.size()[0] as usize
        } else {
            panic!("No track to determine length from")
        }
    }

    pub fn from_pdb<P: AsRef<Path>>(
        path: P,
        chain_id: &str,
        id: Option<&str>,
        is_predicted: bool,
    ) -> Result<Self> {
        let protein_chain = ProteinChain::from_pdb(path, chain_id, id, is_predicted)?;
        Self::from_protein_chain(&protein_chain, false)
    }

    pub fn from_protein_chain(
        protein_chain: &ProteinChain,
        with_annotations: bool,
    ) -> Result<Self> {
        if with_annotations {
            Ok(Self {
                sequence: Some(protein_chain.sequence.clone()),
                secondary_structure: Some(protein_chain.dssp()?.to_string()),
                sasa: Some(protein_chain.sasa()?.into_iter().map(Some).collect()),
                function_annotations: None,
                coordinates: Some(Tensor::new(protein_chain.atom37_positions.clone())?),
                ..Default::default()
            })
        } else {
            Ok(Self {
                sequence: Some(protein_chain.sequence.clone()),
                coordinates: Some(Tensor::new(protein_chain.atom37_positions.clone())?),
                ..Default::default()
            })
        }
    }

    pub fn from_protein_complex(
        protein_complex: &ProteinComplex,
        with_annotations: bool,
    ) -> Result<Self> {
        if with_annotations {
            panic!("Annotations are not supported for ProteinComplex yet")
        }

        Ok(Self {
            sequence: Some(protein_complex.sequence.clone()),
            coordinates: Some(Tensor::new(protein_complex.atom37_positions.clone())?),
            ..Default::default()
        })
    }

    pub fn to_pdb<P: AsRef<Path>>(&self, pdb_path: P) -> Result<()> {
        let complex = self.to_protein_complex(None)?.infer_oxygen()?;
        complex.to_pdb(pdb_path)
    }

    pub fn to_pdb_string(&self) -> Result<String> {
        self.to_protein_chain()?.to_pdb_string()
    }

    pub fn to_protein_chain(&self) -> Result<ProteinChain> {
        let coordinates = self
            .coordinates
            .as_ref()
            .ok_or_else(|| format_err!("Coordinates required"))?;

        ProteinChain::from_atom37(
            coordinates.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
            None,
            self.sequence.as_ref().map(|s| s.replace("_", "X")),
            None,
            None,
            None,
            None,
            self.plddt
                .as_ref()
                .map(|t| t.to_vec1::<f32>())
                .transpose()?,
        )
    }

    pub fn to_protein_complex(
        &self,
        copy_annotations: Option<&ProteinComplex>,
    ) -> Result<ProteinComplex> {
        let sequence = self
            .sequence
            .as_ref()
            .ok_or_else(|| format_err!("Sequence required"))?;
        let coordinates = self
            .coordinates
            .as_ref()
            .ok_or_else(|| format_err!("Coordinates required"))?;

        let coords = coordinates.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        let chain_boundaries = get_chainbreak_boundaries_from_sequence(sequence);
        let gt_chains = copy_annotations.map(|pc| pc.chain_iter().collect::<Vec<_>>());

        let mut pred_chains = Vec::new();
        for (i, (start, end)) in chain_boundaries.iter().enumerate() {
            let chain = ProteinChain::from_atom37(
                coords[*start..*end].to_vec(),
                Some(&sequence[*start..*end]),
                gt_chains
                    .as_ref()
                    .and_then(|chains| chains[i].chain_id.clone()),
                gt_chains.as_ref().and_then(|chains| chains[i].entity_id),
            )?;
            pred_chains.push(chain);
        }

        ProteinComplex::from_chains(pred_chains)
    }
}

#[derive(Debug, Default)]
pub struct ESMProteinTensor {
    sequence: Option<Tensor>,
    structure: Option<Tensor>,
    secondary_structure: Option<Tensor>,
    sasa: Option<Tensor>,
    function: Option<Tensor>,
    residue_annotations: Option<Tensor>,
    coordinates: Option<Tensor>,

    potential_sequence_of_concern: bool,
}

impl ESMProteinTensor {
    fn detect_attribute<F, T>(&self, f: F, msg: &str) -> Option<T>
    where
        F: Fn(&str, &Tensor) -> T,
        T: PartialEq,
    {
        let mut values = Vec::new();

        for (name, tensor) in [
            ("sequence", &self.sequence),
            ("structure", &self.structure),
            ("secondary_structure", &self.secondary_structure),
            ("sasa", &self.sasa),
            ("function", &self.function),
            ("residue_annotations", &self.residue_annotations),
            ("coordinates", &self.coordinates),
        ] {
            if let Some(t) = tensor {
                values.push(f(name, t));
            }
        }

        if values.is_empty() {
            None
        } else {
            let first = values[0];
            if values.iter().all(|v| *v == first) {
                Some(first)
            } else {
                panic!("Inconsistent {}: {:?}", msg, values)
            }
        }
    }

    pub fn len(&self) -> usize {
        self.detect_attribute(|_, t| t.size()[0], "length")
            .unwrap_or(0) as usize
    }

    pub fn device(&self) -> Option<Device> {
        self.detect_attribute(|_, t| t.device(), "device")
    }

    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let mut new = Self::default();

        if let Some(t) = &self.sequence {
            new.sequence = Some(t.to_device(device)?);
        }
        if let Some(t) = &self.structure {
            new.structure = Some(t.to_device(device)?);
        }
        // etc for other fields

        Ok(new)
    }

    pub fn empty(
        length: usize,
        tokenizers: Option<&TokenizerCollectionProtocol>,
        device: &Device,
    ) -> Result<Self> {
        let tokenizers = tokenizers.unwrap_or_else(|| get_model_tokenizers(ESM3_OPEN_SMALL));

        Ok(Self {
            sequence: Some(
                encoding::get_default_sequence_tokens(length, tokenizers.sequence)?
                    .to_device(device)?,
            ),
            structure: Some(
                encoding::get_default_structure_tokens(length, tokenizers.structure)?
                    .to_device(device)?,
            ),
            // etc for other fields
            ..Default::default()
        })
    }
}

#[derive(Debug)]
pub struct ESMProteinError {
    error_code: i32,
    error_msg: String,
}

#[derive(Debug, Default)]
pub struct GenerationConfig {
    pub track: String,
    pub invalid_ids: Vec<i32>,
    pub schedule: String, // "cosine" or "linear"
    pub strategy: String, // "random" or "entropy"
    pub num_steps: i32,
    pub temperature: f32,
    pub temperature_annealing: bool,
    pub top_p: f32,
    pub condition_on_coordinates_only: bool,
}

impl GenerationConfig {
    pub fn use_entropy_based_unmasking_strategy(&mut self) {
        self.schedule = "cosine".to_string();
        self.strategy = "entropy".to_string();
        self.temperature_annealing = false;
    }

    pub fn use_generative_unmasking_strategy(&mut self) {
        self.schedule = "cosine".to_string();
        self.strategy = "random".to_string();
        self.temperature_annealing = true;
    }
}

#[derive(Debug, Default)]
pub struct InverseFoldingConfig {
    pub invalid_ids: Vec<i32>,
    pub temperature: f32,
}

#[derive(Debug, Default)]
pub struct SamplingTrackConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub only_sample_masked_tokens: bool,
    pub invalid_ids: Vec<i32>,
    pub topk_logprobs: i32,
}

#[derive(Debug, Default)]
pub struct SamplingConfig {
    pub sequence: Option<SamplingTrackConfig>,
    pub structure: Option<SamplingTrackConfig>,
    pub secondary_structure: Option<SamplingTrackConfig>,
    pub sasa: Option<SamplingTrackConfig>,
    pub function: Option<SamplingTrackConfig>,

    pub return_per_residue_embeddings: bool,
    pub return_mean_embedding: bool,
}

#[derive(Debug, Default)]
pub struct ForwardTrackData {
    pub sequence: Option<Tensor>,
    pub structure: Option<Tensor>,
    pub secondary_structure: Option<Tensor>,
    pub sasa: Option<Tensor>,
    pub function: Option<Tensor>,
}

#[derive(Debug, Default)]
pub struct LogitsConfig {
    pub sequence: bool,
    pub structure: bool,
    pub secondary_structure: bool,
    pub sasa: bool,
    pub function: bool,
    pub residue_annotations: bool,

    pub return_embeddings: bool,
}

#[derive(Debug, Default)]
pub struct LogitsOutput {
    pub logits: Option<ForwardTrackData>,
    pub embeddings: Option<Tensor>,
    pub residue_annotation_logits: Option<Tensor>,
}

#[derive(Debug)]
pub struct ForwardAndSampleOutput {
    pub logits: Option<ForwardTrackData>,
    pub embeddings: Option<Tensor>,
    pub residue_annotation_logits: Option<Tensor>,
    pub protein_tensor: ESMProteinTensor,
    pub entropy: Option<ForwardTrackData>,
    pub prob: Option<ForwardTrackData>,
    pub logprob: Option<ForwardTrackData>,
    pub top_prob: Option<ForwardTrackData>,
    pub topk_logprob: Option<ForwardTrackData>,
    pub topk_tokens: Option<ForwardTrackData>,
    pub per_residue_embedding: Option<Tensor>,
    pub mean_embedding: Option<Tensor>,
}

pub trait ESM3InferenceClient {
    fn generate(
        &self,
        input: Box<dyn ProteinType>,
        config: GenerationConfig,
    ) -> Result<Box<dyn ProteinType>>;

    fn batch_generate(
        &self,
        inputs: Vec<Box<dyn ProteinType>>,
        configs: Vec<GenerationConfig>,
    ) -> Result<Vec<Box<dyn ProteinType>>>;

    fn encode(&self, input: &ESMProtein) -> Result<ESMProteinTensor>;

    fn decode(&self, input: &ESMProteinTensor) -> Result<ESMProtein>;

    fn logits(&self, input: &ESMProteinTensor, config: LogitsConfig) -> Result<LogitsOutput>;

    fn forward_and_sample(
        &self,
        input: &ESMProteinTensor,
        config: SamplingConfig,
    ) -> Result<ForwardAndSampleOutput>;

    fn raw_model(&self) -> &dyn std::any::Any;
}

pub trait ESMCInferenceClient {
    fn encode(&self, input: &ESMProtein) -> Result<ESMProteinTensor>;

    fn decode(&self, input: &ESMProteinTensor) -> Result<ESMProtein>;

    fn logits(&self, input: &ESMProteinTensor, config: LogitsConfig) -> Result<LogitsOutput>;

    fn raw_model(&self) -> &dyn std::any::Any;
}
