//! ProtT5 runner — loads `Rostlab/prot_t5_xl_half_uniref50-enc` and embeds
//! sequences through `candle_transformers::models::t5::T5EncoderModel`.

use crate::loader::{LoadOptions, WeightSource};
use crate::plm_runner::{ModelMetadata, PlmRunner, SpecialTokenLayout};
use crate::prott5::tokenizer;
use crate::registry::{self, ModelCard};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use std::sync::Mutex;

/// Available ProtT5 variants.
pub enum ProtT5Models {
    /// `prot_t5_xl_half_uniref50-enc` — the encoder-only half-precision build,
    /// which is the one almost everyone uses. The full encoder-decoder
    /// `prot_t5_xl_uniref50` doubles the download for a decoder that
    /// embedding work never runs.
    XlHalfUniref50Enc,
}

impl ProtT5Models {
    /// This variant's registry id.
    pub const fn registry_id(&self) -> &'static str {
        match self {
            Self::XlHalfUniref50Enc => "prott5-xl-half-uniref50-enc",
        }
    }

    /// This variant's [`ModelCard`].
    pub fn card(&self) -> &'static ModelCard {
        registry::lookup(self.registry_id())
            .expect("every ProtT5Models variant must have a registry entry")
    }

    /// Where this variant's weights live, plus its built-in fallback config.
    pub fn model_info(&self) -> (WeightSource, &'static str, T5Config) {
        let card = self.card();
        (card.source, card.file, xl_config())
    }
}

/// The published `config.json` for ProtT5-XL, as a fallback.
///
/// Used only when the hub's `config.json` is unreachable or unparseable; the
/// values are copied from the checkpoint's own file.
fn xl_config() -> T5Config {
    T5Config {
        vocab_size: tokenizer::VOCAB_SIZE,
        d_model: 1024,
        d_kv: 128,
        d_ff: 16384,
        num_layers: 24,
        num_decoder_layers: Some(24),
        num_heads: 32,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        dropout_rate: 0.1,
        layer_norm_epsilon: 1e-6,
        initializer_factor: 1.0,
        feed_forward_proj: Default::default(),
        tie_word_embeddings: false,
        is_decoder: false,
        is_encoder_decoder: true,
        use_cache: false,
        pad_token_id: tokenizer::PAD_ID as usize,
        eos_token_id: tokenizer::EOS_ID as usize,
        decoder_start_token_id: Some(tokenizer::PAD_ID as usize),
    }
}

/// ProtT5 encoder wrapped for [`PlmRunner`].
///
/// # Why the model sits behind a `Mutex`
///
/// `T5EncoderModel::forward` takes `&mut self`, because the same `T5Attention`
/// code serves the decoder, where it maintains a KV cache. [`PlmRunner::embed`]
/// takes `&self`, and widening the trait to `&mut self` would make every
/// runner harder to share across threads for the sake of one model.
///
/// The encoder does not actually cache. `T5Attention::load` sets
/// `use_cache: cfg.use_cache && decoder`, and `T5EncoderModel::load` builds its
/// stack with `decoder = false`, so the cache branch is dead on this path
/// whatever `config.json` says. `load_config` also forces `use_cache: false`
/// so the guarantee does not depend on a detail of candle's internals, and
/// `test_repeated_embed_is_deterministic` fails if a future candle version
/// starts carrying state across calls.
///
/// So the `Mutex` buys the `&mut` the signature demands while keeping the
/// runner `Send + Sync`, and it is uncontended in the sense that matters: there
/// is no state to protect. It does serialise concurrent `embed` calls on one
/// runner, which is a real cost — clone the runner per thread if that bites.
pub struct ProtT5Runner {
    model: Mutex<T5EncoderModel>,
    config: T5Config,
    device: Device,
}

impl ProtT5Runner {
    /// Load from the hub at the checkpoint's own precision (F16).
    ///
    /// The weights are published as `float16` and the model is 2.4 GB on disk.
    /// Loading it at F32 would double that to ~4.8 GB in memory for no gain in
    /// fidelity — the stored values are F16 either way — so F16 is the default
    /// here rather than the crate-wide F32 (ferritin-100.9). Use
    /// [`from_pretrained_with`][Self::from_pretrained_with] to override.
    pub fn from_pretrained(model: ProtT5Models, device: Device) -> Result<Self> {
        Self::from_pretrained_with(model, &LoadOptions::new(device).with_dtype(DType::F16))
    }

    /// Load with an explicit device and dtype.
    pub fn from_pretrained_with(model: ProtT5Models, opts: &LoadOptions) -> Result<Self> {
        let (source, file, fallback) = model.model_info();
        let config = load_config(&source, fallback);
        let vb = source.var_builder(file, opts)?;
        let encoder = T5EncoderModel::load(vb, &config)
            .with_context(|| format!("failed to load {} as a T5 encoder", source.repo_id))?;
        Ok(Self {
            model: Mutex::new(encoder),
            config,
            device: opts.device.clone(),
        })
    }

    /// Token ids for `sequence`, shaped `(1, L + 1)` for the trailing `</s>`.
    fn encode(&self, sequence: &str) -> Result<Tensor> {
        let ids = tokenizer::encode(sequence);
        Ok(Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?)
    }
}

/// Load `config.json` from the hub, falling back to the built-in config.
///
/// candle's `t5::Config` derives `Deserialize` over the same field names
/// HuggingFace writes, so the published file parses directly; the extra keys
/// ProtT5 carries (`architectures`, `n_positions`, `torch_dtype`, …) are
/// ignored.
///
/// `use_cache` is forced off afterwards regardless of what the file says —
/// ProtT5's `config.json` sets it to `true`, which is meaningful for the
/// decoder this checkpoint does not contain. See [`ProtT5Runner`].
///
/// The fallback is loud, matching `ESM2Runner::resolve_config`: a config the
/// struct cannot represent silently becoming a different model's config is the
/// failure ferritin-goh.9 was about.
fn load_config(source: &WeightSource, fallback: T5Config) -> T5Config {
    let mut config = match source.fetch_optional("config.json") {
        None => {
            eprintln!(
                "warning: {}: could not download config.json; using the built-in \
                 config for this variant. Verify it matches the checkpoint.",
                source.repo_id
            );
            fallback
        }
        Some(path) => match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<T5Config>(&s).map_err(|e| e.to_string()))
        {
            Ok(config) => config,
            Err(e) => {
                eprintln!(
                    "warning: {}: config.json did not parse as a T5 Config ({e}); \
                     using the built-in config instead. If this checkpoint is not a \
                     T5 variant, the built-in config is probably wrong for it.",
                    source.repo_id
                );
                fallback
            }
        },
    };
    config.use_cache = false;
    config
}

impl PlmRunner for ProtT5Runner {
    /// Run the T5 encoder and return per-token hidden states.
    ///
    /// Shape: `(1, L + 1, d_model)` — the trailing `</s>` row is included, and
    /// there is no BOS row.
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        let ids = self.encode(sequence)?;
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("ProtT5 model mutex was poisoned by an earlier panic"))?;
        Ok(model.forward(&ids)?)
    }

    fn model_name(&self) -> &str {
        "prott5"
    }

    /// ProtT5 appends `</s>` and prepends nothing: T5's
    /// `build_inputs_with_special_tokens` is `ids + [eos]`, and the vocabulary
    /// has no BOS piece at all.
    ///
    /// Every other runner in the crate is `BOS_EOS`, which is exactly why the
    /// layout is declared rather than assumed — reading a ProtT5 embedding as
    /// if row 0 were a BOS token shifts every residue by one.
    fn special_tokens(&self) -> SpecialTokenLayout {
        SpecialTokenLayout::EOS_ONLY
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            d_model: self.config.d_model,
            n_layers: self.config.num_layers,
            vocab_size: self.config.vocab_size,
            // Relative position buckets: no hard architectural cap. The
            // published config carries `n_positions: 512`, but T5 inherits
            // nothing from it — the buckets are computed from the query/key
            // offset at run time.
            max_positions: None,
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    /// Whitespace is not a residue: `"M K T"` is three residues, not five.
    fn residue_count(&self, sequence: &str) -> usize {
        tokenizer::residue_count(sequence)
    }
}
