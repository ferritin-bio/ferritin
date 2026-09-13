//! ProGen2-small, an autoregressive protein language model.
//!
//! The port follows Salesforce's ProGen implementation rather than treating
//! it as a generic GPT-2: its fused projection is laid out as query/value/key
//! in eight model-parallel chunks, and its rotary embedding rotates adjacent
//! pairs in only the first `rotary_dim` channels. Those details are invisible
//! in tensor shapes, but changing either produces plausible-looking wrong
//! scores.

use crate::generative::GenerativeModel;
use crate::loader::{LoadOptions, WeightSource};
use crate::plm_runner::ModelMetadata;
use anyhow::{Context, Result, anyhow, bail};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::{
    Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear, linear_no_bias,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

const QKV_MODEL_PARALLEL_CHUNKS: usize = 8;
const START_TOKEN: u32 = 3; // tokenizer token "1"
const END_TOKEN: u32 = 4; // tokenizer token "2"
const FIRST_RESIDUE_TOKEN: u32 = 5;
const LAST_RESIDUE_TOKEN: u32 = 29;

/// Configuration fields used by the ProGen custom HuggingFace model.
#[derive(Debug, Clone)]
pub struct ProGenConfig {
    pub vocab_size_emb: usize,
    pub vocab_size_lm_head: usize,
    pub n_positions: usize,
    pub embed_dim: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub rotary_dim: Option<usize>,
    pub n_inner: usize,
    pub layer_norm_epsilon: f64,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

#[derive(Debug, Deserialize, Default)]
struct ProGenConfigFile {
    vocab_size: Option<usize>,
    vocab_size_emb: Option<usize>,
    vocab_size_lm_head: Option<usize>,
    n_positions: Option<usize>,
    n_ctx: Option<usize>,
    n_embd: Option<usize>,
    embed_dim: Option<usize>,
    n_layer: Option<usize>,
    n_head: Option<usize>,
    rotary_dim: Option<usize>,
    n_inner: Option<usize>,
    layer_norm_epsilon: Option<f64>,
    activation_function: Option<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl ProGenConfig {
    /// The published `hugohrban/progen2-small` configuration.
    pub fn small() -> Self {
        Self {
            vocab_size_emb: 32,
            vocab_size_lm_head: 32,
            n_positions: 1024,
            embed_dim: 1024,
            n_layer: 12,
            n_head: 16,
            rotary_dim: Some(32),
            n_inner: 4096,
            layer_norm_epsilon: 1e-5,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Parse the model's custom `config.json`, accepting both the original
    /// `n_embd` spelling and the current `embed_dim` spelling.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: ProGenConfigFile =
            serde_json::from_str(json).context("ProGen config.json is not valid JSON")?;
        let vocab_size_emb = raw
            .vocab_size_emb
            .or(raw.vocab_size)
            .ok_or_else(|| anyhow!("ProGen config is missing vocab_size_emb/vocab_size"))?;
        let vocab_size_lm_head = raw
            .vocab_size_lm_head
            .or(raw.vocab_size)
            .unwrap_or(vocab_size_emb);
        let n_positions = raw
            .n_positions
            .or(raw.n_ctx)
            .ok_or_else(|| anyhow!("ProGen config is missing n_positions/n_ctx"))?;
        let embed_dim = raw
            .embed_dim
            .or(raw.n_embd)
            .ok_or_else(|| anyhow!("ProGen config is missing embed_dim/n_embd"))?;
        let n_layer = raw
            .n_layer
            .ok_or_else(|| anyhow!("ProGen config is missing n_layer"))?;
        let n_head = raw
            .n_head
            .ok_or_else(|| anyhow!("ProGen config is missing n_head"))?;
        if let Some(activation) = raw.activation_function.as_deref()
            && activation != "gelu_new"
            && activation != "gelu"
        {
            bail!("unsupported ProGen activation_function {activation:?}");
        }

        let config = Self {
            vocab_size_emb,
            vocab_size_lm_head,
            n_positions,
            embed_dim,
            n_layer,
            n_head,
            rotary_dim: raw.rotary_dim.filter(|dim| *dim > 0),
            n_inner: raw.n_inner.unwrap_or(embed_dim * 4),
            layer_norm_epsilon: raw.layer_norm_epsilon.unwrap_or(1e-5),
            bos_token_id: raw.bos_token_id.unwrap_or(1),
            eos_token_id: raw.eos_token_id.unwrap_or(2),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.vocab_size_emb == 0 || self.vocab_size_lm_head == 0 {
            bail!("ProGen vocabulary sizes must be non-zero");
        }
        if self.n_positions == 0 || self.n_layer == 0 || self.n_head == 0 {
            bail!("ProGen positions, layers, and heads must be non-zero");
        }
        if !self.embed_dim.is_multiple_of(self.n_head) {
            bail!(
                "ProGen embed_dim {} is not divisible by n_head {}",
                self.embed_dim,
                self.n_head
            );
        }
        if !self.n_head.is_multiple_of(QKV_MODEL_PARALLEL_CHUNKS) {
            bail!(
                "ProGen n_head {} must be divisible by {} to match the checkpoint's fused QKV layout",
                self.n_head,
                QKV_MODEL_PARALLEL_CHUNKS
            );
        }
        if let Some(rotary_dim) = self.rotary_dim {
            let head_dim = self.embed_dim / self.n_head;
            if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
                bail!(
                    "ProGen rotary_dim {} must be even and at most head_dim {}",
                    rotary_dim,
                    head_dim
                );
            }
        }
        Ok(())
    }

    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            d_model: self.embed_dim,
            n_layers: self.n_layer,
            vocab_size: self.vocab_size_lm_head,
            max_positions: Some(self.n_positions),
        }
    }
}

/// ProGen checkpoints available through the runner.
#[derive(Debug, Clone, Copy)]
pub enum ProGenModels {
    /// `hugohrban/progen2-small`, 151M parameters.
    Small,
}

impl ProGenModels {
    pub const fn registry_id(&self) -> &'static str {
        match self {
            Self::Small => "progen2-small",
        }
    }

    pub fn model_info(&self) -> (WeightSource, ProGenConfig) {
        match self {
            Self::Small => (
                WeightSource::safetensors("hugohrban/progen2-small").at_revision("main"),
                ProGenConfig::small(),
            ),
        }
    }
}

struct RotaryEmbedding {
    rotary_dim: usize,
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(rotary_dim: usize, max_positions: usize, device: &Device) -> Result<Self> {
        let inv_freq = Tensor::from_iter(
            (0..rotary_dim / 2)
                .map(|i| 10000f64.powf(-((2 * i) as f64) / rotary_dim as f64) as f32),
            device,
        )?;
        let positions = Tensor::arange(0u32, max_positions as u32, device)?.to_dtype(DType::F32)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        // ProGen rotates adjacent pairs: [x0, x1] -> [-x1, x0]. Repeat
        // each frequency twice, rather than concatenating the frequency table
        // with itself as a non-interleaved RoPE implementation would do.
        let cos = freqs
            .cos()?
            .unsqueeze(D::Minus1)?
            .repeat((1, 1, 2))?
            .reshape((1, 1, max_positions, rotary_dim))?;
        let sin = freqs
            .sin()?
            .unsqueeze(D::Minus1)?
            .repeat((1, 1, 2))?
            .reshape((1, 1, max_positions, rotary_dim))?;
        Ok(Self {
            rotary_dim,
            cos,
            sin,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, heads, seq, head_dim) = x.dims4()?;
        let x_rot = x.narrow(D::Minus1, 0, self.rotary_dim)?;
        let x_pass = x.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let pairs = x_rot.reshape((batch, heads, seq, self.rotary_dim / 2, 2))?;
        let even = pairs.i((.., .., .., .., 0))?;
        let odd = pairs.i((.., .., .., .., 1))?;
        let rotated = Tensor::stack(&[&odd.neg()?, &even], D::Minus1)?.flatten_from(D::Minus2)?;
        let cos = self.cos.narrow(2, 0, seq)?;
        let sin = self.sin.narrow(2, 0, seq)?;
        let x_rot = x_rot
            .broadcast_mul(&cos)?
            .add(&rotated.broadcast_mul(&sin)?)?;
        Ok(Tensor::cat(&[&x_rot, &x_pass], D::Minus1)?)
    }
}

struct ProGenAttention {
    qkv_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    rotary: Option<RotaryEmbedding>,
    causal_mask: Tensor,
}

impl ProGenAttention {
    fn load(vb: VarBuilder, config: &ProGenConfig) -> Result<Self> {
        let head_dim = config.embed_dim / config.n_head;
        let qkv_proj = linear_no_bias(config.embed_dim, config.embed_dim * 3, vb.pp("qkv_proj"))?;
        let out_proj = linear_no_bias(config.embed_dim, config.embed_dim, vb.pp("out_proj"))?;
        let mut mask = vec![0f32; config.n_positions * config.n_positions];
        for query in 0..config.n_positions {
            for key in query + 1..config.n_positions {
                mask[query * config.n_positions + key] = -1e9;
            }
        }
        let causal_mask = Tensor::from_vec(
            mask,
            (1, 1, config.n_positions, config.n_positions),
            vb.device(),
        )?;
        let rotary = config
            .rotary_dim
            .map(|dim| RotaryEmbedding::new(dim, config.n_positions, vb.device()))
            .transpose()?;
        Ok(Self {
            qkv_proj,
            out_proj,
            num_heads: config.n_head,
            head_dim,
            rotary,
            causal_mask,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = hidden_states.dims3()?;
        let qkv = self.qkv_proj.forward(hidden_states)?;
        let local_dim = self.head_dim * self.num_heads / QKV_MODEL_PARALLEL_CHUNKS;
        let qkv = qkv.reshape((batch, seq, QKV_MODEL_PARALLEL_CHUNKS, local_dim * 3))?;

        // The upstream checkpoint stores Q, V, K in each of eight logical
        // model-parallel chunks. Do not split the fused projection into three
        // contiguous embed-sized pieces: that silently scrambles the heads.
        let query = qkv
            .narrow(D::Minus1, 0, local_dim)?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let value = qkv
            .narrow(D::Minus1, local_dim, local_dim)?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let key = qkv
            .narrow(D::Minus1, local_dim * 2, local_dim)?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let (query, key) = match &self.rotary {
            Some(rotary) => (rotary.forward(&query)?, rotary.forward(&key)?),
            None => (query, key),
        };
        let scores = query
            .matmul(&key.transpose(2, 3)?)?
            .affine((self.head_dim as f64).sqrt().recip(), 0.0)?;
        let mask = self.causal_mask.narrow(2, 0, seq)?.narrow(3, 0, seq)?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = softmax(&scores, D::Minus1)?.to_dtype(value.dtype())?;
        let attended = weights
            .matmul(&value)?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, seq, self.num_heads * self.head_dim))?;
        Ok(self.out_proj.forward(&attended)?)
    }
}

struct ProGenMlp {
    fc_in: Linear,
    fc_out: Linear,
}

impl ProGenMlp {
    fn load(vb: VarBuilder, config: &ProGenConfig) -> Result<Self> {
        Ok(Self {
            fc_in: linear(config.embed_dim, config.n_inner, vb.pp("mlp.fc_in"))?,
            fc_out: linear(config.n_inner, config.embed_dim, vb.pp("mlp.fc_out"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // `gelu_new` is the tanh approximation, which is candle's `gelu()`;
        // candle's exact `gelu_erf()` would not match the checkpoint.
        Ok(self
            .fc_out
            .forward(&self.fc_in.forward(hidden_states)?.gelu()?)?)
    }
}

struct ProGenBlock {
    ln_1: LayerNorm,
    attention: ProGenAttention,
    mlp: ProGenMlp,
}

impl ProGenBlock {
    fn load(vb: VarBuilder, config: &ProGenConfig) -> Result<Self> {
        Ok(Self {
            ln_1: layer_norm(config.embed_dim, config.layer_norm_epsilon, vb.pp("ln_1"))?,
            attention: ProGenAttention::load(vb.pp("attn"), config)?,
            mlp: ProGenMlp::load(vb, config)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let normalized = self.ln_1.forward(hidden_states)?;
        let attention = self.attention.forward(&normalized)?;
        let feed_forward = self.mlp.forward(&normalized)?;
        Ok(hidden_states
            .broadcast_add(&attention)?
            .broadcast_add(&feed_forward)?)
    }
}

struct ProGenForCausalLm {
    embeddings: Embedding,
    blocks: Vec<ProGenBlock>,
    final_layer_norm: LayerNorm,
    lm_head: Linear,
}

impl ProGenForCausalLm {
    fn load(vb: VarBuilder, config: &ProGenConfig) -> Result<Self> {
        let embeddings = embedding(
            config.vocab_size_emb,
            config.embed_dim,
            vb.pp("transformer.wte"),
        )?;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            blocks.push(ProGenBlock::load(
                vb.pp(format!("transformer.h.{layer}")),
                config,
            )?);
        }
        let final_layer_norm = layer_norm(
            config.embed_dim,
            config.layer_norm_epsilon,
            vb.pp("transformer.ln_f"),
        )?;
        let lm_head = linear(
            config.embed_dim,
            config.vocab_size_lm_head,
            vb.pp("lm_head"),
        )?;
        Ok(Self {
            embeddings,
            blocks,
            final_layer_norm,
            lm_head,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }
        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        Ok(self.lm_head.forward(&hidden_states)?)
    }
}

/// ProGen2-small runner for deterministic sequence scoring.
pub struct ProGen2 {
    model: ProGenForCausalLm,
    tokenizer: Tokenizer,
    config: ProGenConfig,
    device: Device,
    name: &'static str,
}

impl ProGen2 {
    /// Load ProGen2-small from the HuggingFace hub at F32.
    pub fn from_pretrained(model: ProGenModels, device: Device) -> Result<Self> {
        Self::from_pretrained_with(model, &LoadOptions::new(device))
    }

    /// Load ProGen2-small with an explicit dtype/device.
    pub fn from_pretrained_with(model: ProGenModels, opts: &LoadOptions) -> Result<Self> {
        let (source, fallback) = model.model_info();
        let config = resolve_config(&source, fallback)?;
        let vb = source.var_builder("model.safetensors", opts)?;
        Self::from_var_builder_with_name(config, vb, opts.device.clone(), model.registry_id())
    }

    /// Construct a runner from a caller-provided builder.
    ///
    /// This is useful for tiny architecture tests and keeps those tests
    /// independent of a 617 MB checkpoint download.
    pub fn from_var_builder(config: ProGenConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        Self::from_var_builder_with_name(config, vb, device, "progen2-small")
    }

    fn from_var_builder_with_name(
        config: ProGenConfig,
        vb: VarBuilder,
        device: Device,
        name: &'static str,
    ) -> Result<Self> {
        config.validate()?;
        let model = ProGenForCausalLm::load(vb, &config)?;
        let tokenizer = Tokenizer::from_bytes(include_bytes!("../progen/tokenizer.json"))
            .map_err(|e| anyhow!("failed to load embedded ProGen tokenizer: {e}"))?;
        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            name,
        })
    }

    /// Run the causal model on a protein sequence, including `1` and `2`.
    pub fn run_forward(&self, sequence: &str) -> Result<Tensor> {
        let ids = self.token_ids(sequence)?;
        let input_ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
        Ok(self.model.forward(&input_ids)?.to_dtype(DType::F32)?)
    }
}

impl GenerativeModel for ProGen2 {
    fn model_name(&self) -> &str {
        self.name
    }

    fn metadata(&self) -> ModelMetadata {
        self.config.metadata()
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn token_ids(&self, sequence: &str) -> Result<Vec<u32>> {
        if sequence.is_empty() {
            bail!("{}: cannot score an empty sequence", self.name);
        }
        let wrapped = format!("1{sequence}2");
        let encoded = self
            .tokenizer
            .encode(wrapped, false)
            .map_err(|e| anyhow!("{}: tokenizer failed: {e}", self.name))?;
        let ids = encoded.get_ids().to_vec();
        if ids.first().copied() != Some(START_TOKEN) || ids.last().copied() != Some(END_TOKEN) {
            bail!(
                "{}: tokenizer did not preserve ProGen terminal tokens; got {:?}",
                self.name,
                ids
            );
        }
        if ids[1..ids.len() - 1]
            .iter()
            .any(|id| !(*id >= FIRST_RESIDUE_TOKEN && *id <= LAST_RESIDUE_TOKEN))
        {
            bail!(
                "{}: sequence contains a residue outside ProGen's supported alphabet A-Z without J",
                self.name
            );
        }
        if ids.len() > self.config.n_positions {
            bail!(
                "{}: sequence encodes to {} tokens, exceeding the {}-token context window",
                self.name,
                ids.len(),
                self.config.n_positions
            );
        }
        Ok(ids)
    }

    fn logits(&self, sequence: &str) -> Result<Tensor> {
        self.run_forward(sequence)
    }

    fn log_likelihood(&self, sequence: &str) -> Result<f32> {
        let ids = self.token_ids(sequence)?;
        let residue_count = ids.len() - 2;
        let logits = self.logits(sequence)?.to_dtype(DType::F32)?;
        let residue_logits = logits.i((
            0,
            0..residue_count,
            FIRST_RESIDUE_TOKEN as usize..(LAST_RESIDUE_TOKEN as usize + 1),
        ))?;
        let log_probs = log_softmax(&residue_logits, D::Minus1)?;
        let targets = Tensor::from_vec(
            ids[1..=residue_count]
                .iter()
                .map(|id| id - FIRST_RESIDUE_TOKEN)
                .collect(),
            (residue_count, 1),
            logits.device(),
        )?;
        Ok(log_probs
            .gather(&targets, D::Minus1)?
            .squeeze(D::Minus1)?
            .sum_all()?
            .to_scalar::<f32>()?)
    }
}

fn resolve_config(source: &WeightSource, fallback: ProGenConfig) -> Result<ProGenConfig> {
    let Some(path) = source.fetch_optional("config.json") else {
        eprintln!(
            "warning: {}: could not download config.json; using the built-in ProGen-small config",
            source.repo_id
        );
        return Ok(fallback);
    };
    let config_json = std::fs::read_to_string(path)?;
    match ProGenConfig::from_json(&config_json) {
        Ok(config) => Ok(config),
        Err(error) => {
            eprintln!(
                "warning: {}: config.json did not parse as ProGenConfig ({error}); using the built-in config",
                source.repo_id
            );
            Ok(fallback)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generative::GenerativeModel;
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_runner() -> Result<ProGen2> {
        let config = ProGenConfig {
            vocab_size_emb: 32,
            vocab_size_lm_head: 32,
            n_positions: 32,
            embed_dim: 64,
            n_layer: 2,
            n_head: 8,
            rotary_dim: Some(8),
            n_inner: 128,
            layer_norm_epsilon: 1e-5,
            bos_token_id: 1,
            eos_token_id: 2,
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        ProGen2::from_var_builder(config, vb, device)
    }

    #[test]
    fn test_small_config_matches_published_dimensions() {
        let config = ProGenConfig::small();
        assert_eq!(config.metadata().d_model, 1024);
        assert_eq!(config.metadata().n_layers, 12);
        assert_eq!(config.metadata().vocab_size, 32);
        assert_eq!(config.metadata().max_positions, Some(1024));
    }

    #[test]
    fn test_config_parser_accepts_published_field_names() -> Result<()> {
        let config = ProGenConfig::from_json(
            r#"{
                "vocab_size_emb": 32,
                "vocab_size_lm_head": 32,
                "n_positions": 1024,
                "embed_dim": 1024,
                "n_layer": 12,
                "n_head": 16,
                "rotary_dim": 32,
                "activation_function": "gelu_new"
            }"#,
        )?;
        assert_eq!(config.metadata(), ProGenConfig::small().metadata());
        Ok(())
    }

    #[test]
    fn test_tokenizer_wraps_terminal_tokens() -> Result<()> {
        let runner = tiny_runner()?;
        assert_eq!(runner.token_ids("ACD")?, vec![3, 5, 7, 8, 4]);
        Ok(())
    }

    #[test]
    fn test_causal_logits_do_not_see_future_residues() -> Result<()> {
        let runner = tiny_runner()?;
        let short = runner.logits("ACD")?.i((0, 0..4, ..))?;
        let extended = runner.logits("ACDD")?.i((0, 0..4, ..))?;
        let max_diff = (&short - &extended)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            max_diff < 1e-6,
            "future residue leaked into causal logits: {max_diff}"
        );
        Ok(())
    }

    #[test]
    fn test_log_likelihood_is_finite_and_length_normalizable() -> Result<()> {
        let runner = tiny_runner()?;
        let sum = runner.log_likelihood("ACD")?;
        let mean = runner.mean_log_likelihood("ACD")?;
        assert!(sum.is_finite() && mean.is_finite());
        assert!((sum / 3.0 - mean).abs() < 1e-5);
        Ok(())
    }
}
