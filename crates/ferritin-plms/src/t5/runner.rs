//! T5-family runner — embeds sequences through
//! `candle_transformers::models::t5::T5EncoderModel`.
//!
//! One runner serves ProtT5 and Ankh because they differ in nothing this code
//! cares about: same architecture, same residue alphabet and ids, same
//! `EOS_ONLY` token layout. What differs is dimensions and the FFN activation
//! (Ankh is gated-gelu), and both of those come from the checkpoint's own
//! `config.json`.

use crate::loader::{LoadOptions, WeightSource};
use crate::plm_runner::{ModelMetadata, PlmRunner, SpecialTokenLayout};
use crate::registry::{self, ModelCard};
use crate::t5::tokenizer;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5::{
    ActivationWithOptionalGating, Config as T5Config, T5EncoderModel,
};
use std::sync::Mutex;

/// Available T5-family models.
pub enum T5Models {
    /// `Rostlab/prot_t5_xl_half_uniref50-enc` — the encoder-only
    /// half-precision build, which is the one almost everyone uses. The full
    /// encoder-decoder `prot_t5_xl_uniref50` doubles the download for a
    /// decoder that embedding work never runs.
    ProtT5XlHalfUniref50Enc,
    /// `ElnaggarLab/ankh-base` — 48 encoder layers at `d_model` 768.
    AnkhBase,
    /// `ElnaggarLab/ankh-large` — 48 encoder layers at `d_model` 1536.
    AnkhLarge,
}

impl T5Models {
    /// This variant's registry id.
    pub const fn registry_id(&self) -> &'static str {
        match self {
            Self::ProtT5XlHalfUniref50Enc => "prott5-xl-half-uniref50-enc",
            Self::AnkhBase => "ankh-base",
            Self::AnkhLarge => "ankh-large",
        }
    }

    /// This variant's [`ModelCard`].
    pub fn card(&self) -> &'static ModelCard {
        registry::lookup(self.registry_id())
            .expect("every T5Models variant must have a registry entry")
    }

    /// The precision this checkpoint is published at.
    ///
    /// ProtT5's encoder-only build ships `float16`, so loading it at F32 would
    /// double 2.4 GB in memory for values that are F16 either way. Ankh ships
    /// `float32`, and rounding it down would lose real precision rather than
    /// save anything the checkpoint has. Either can be overridden through
    /// [`T5Runner::from_pretrained_with`] (ferritin-100.9).
    pub const fn published_dtype(&self) -> DType {
        match self {
            Self::ProtT5XlHalfUniref50Enc => DType::F16,
            Self::AnkhBase | Self::AnkhLarge => DType::F32,
        }
    }

    /// Where this variant's weights live, plus its built-in fallback config.
    pub fn model_info(&self) -> (WeightSource, &'static str, T5Config) {
        let card = self.card();
        (card.source, card.file, self.fallback_config())
    }

    /// The published `config.json`, transcribed, for when the hub is
    /// unreachable. Values copied from each checkpoint's own file.
    fn fallback_config(&self) -> T5Config {
        let card = self.card();
        let base = T5Config {
            vocab_size: card.metadata.vocab_size,
            d_model: card.metadata.d_model,
            num_layers: card.metadata.n_layers,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            initializer_factor: 1.0,
            layer_norm_epsilon: 1e-6,
            tie_word_embeddings: false,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: false,
            pad_token_id: tokenizer::PAD_ID as usize,
            eos_token_id: tokenizer::EOS_ID as usize,
            decoder_start_token_id: Some(tokenizer::PAD_ID as usize),
            // Overridden per model below.
            d_kv: 64,
            d_ff: 0,
            num_decoder_layers: None,
            num_heads: 0,
            dropout_rate: 0.0,
            feed_forward_proj: Default::default(),
        };
        match self {
            Self::ProtT5XlHalfUniref50Enc => T5Config {
                d_kv: 128,
                d_ff: 16384,
                num_decoder_layers: Some(24),
                num_heads: 32,
                dropout_rate: 0.1,
                ..base
            },
            // Ankh is gated: `feed_forward_proj: "gated-gelu"`, which candle
            // parses into a gated `T5DenseGatedActDense` rather than the plain
            // ReLU dense ProtT5 uses.
            Self::AnkhBase => T5Config {
                d_kv: 64,
                d_ff: 3072,
                num_decoder_layers: Some(24),
                num_heads: 12,
                feed_forward_proj: gated_gelu(),
                ..base
            },
            Self::AnkhLarge => T5Config {
                d_kv: 64,
                d_ff: 3840,
                num_decoder_layers: Some(24),
                num_heads: 16,
                feed_forward_proj: gated_gelu(),
                ..base
            },
        }
    }
}

/// Ankh's `"gated-gelu"` feed-forward.
///
/// Built as a struct literal rather than by deserializing the string: candle
/// maps `"gated-gelu"` through `deserialize_feed_forward_proj_activation`, a
/// *field-level* `deserialize_with` on `Config`, so
/// `ActivationWithOptionalGating`'s own derived `Deserialize` expects a struct
/// and rejects the bare string. This mirrors what that function produces —
/// `NewGelu`, i.e. the tanh approximation, matching Ankh's
/// `dense_act_fn: "gelu_new"`.
fn gated_gelu() -> ActivationWithOptionalGating {
    ActivationWithOptionalGating {
        gated: true,
        activation: candle_nn::Activation::NewGelu,
    }
}

/// A T5 encoder wrapped for [`PlmRunner`].
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
pub struct T5Runner {
    model: Mutex<T5EncoderModel>,
    config: T5Config,
    device: Device,
    /// The registry id, so `model_name` distinguishes ProtT5 from Ankh in
    /// error messages and conformance output.
    name: &'static str,
}

impl T5Runner {
    /// Load from the hub at the checkpoint's own published precision.
    ///
    /// See [`T5Models::published_dtype`]: F16 for ProtT5, F32 for Ankh. Use
    /// [`from_pretrained_with`][Self::from_pretrained_with] to override.
    pub fn from_pretrained(model: T5Models, device: Device) -> Result<Self> {
        let dtype = model.published_dtype();
        Self::from_pretrained_with(model, &LoadOptions::new(device).with_dtype(dtype))
    }

    /// Load with an explicit device and dtype.
    pub fn from_pretrained_with(model: T5Models, opts: &LoadOptions) -> Result<Self> {
        let (source, file, fallback) = model.model_info();
        let config = load_config(&source, fallback);
        let vb = source.var_builder(file, opts)?;
        let encoder = T5EncoderModel::load(vb, &config)
            .with_context(|| format!("failed to load {} as a T5 encoder", source.repo_id))?;
        Ok(Self {
            model: Mutex::new(encoder),
            config,
            device: opts.device.clone(),
            name: model.registry_id(),
        })
    }

    /// Token ids for `sequence`, shaped `(1, L + 1)` for the trailing `</s>`.
    fn encode(&self, sequence: &str) -> Result<Tensor> {
        let ids = tokenizer::encode(sequence);
        Ok(Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?)
    }
}

/// Parse a HuggingFace T5 `config.json` into candle's `Config`.
///
/// Strips keys whose value is `null` before deserializing. That is not
/// cosmetic: candle marks optional fields `#[serde(default = "...")]`, and
/// serde's `default` fills in a **missing** key, not a present-but-null one. So
/// `"tie_word_embeddings": null` — which is exactly what
/// `Rostlab/ProstT5_fp16` ships — fails with *"invalid type: null, expected a
/// boolean"* and drops the whole config on the floor.
///
/// For ProstT5 that would be quietly serious rather than merely noisy:
/// `tie_word_embeddings` decides whether `decode` scales the hidden state by
/// `sqrt(d_model)` and projects through the shared embedding, and the
/// checkpoint has no `lm_head` to fall back on.
///
/// Treating an explicit null as absent is what HuggingFace itself means by it.
pub(crate) fn parse_t5_config(json: &str) -> Result<T5Config> {
    let mut value: serde_json::Value = serde_json::from_str(json)?;
    if let Some(map) = value.as_object_mut() {
        map.retain(|_, v| !v.is_null());
    }
    Ok(serde_json::from_value(value)?)
}

/// Load `config.json` from the hub, falling back to the built-in config.
///
/// candle's `t5::Config` derives `Deserialize` over the same field names
/// HuggingFace writes, so the published file parses directly once nulls are
/// stripped; the extra keys these repos carry (`architectures`, `n_positions`,
/// `torch_dtype`, …) are ignored.
///
/// `use_cache` is forced off afterwards regardless of what the file says — it
/// is meaningful for a decoder, and this loads an encoder. See [`T5Runner`].
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
            .and_then(|s| parse_t5_config(&s).map_err(|e| e.to_string()))
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

impl PlmRunner for T5Runner {
    /// Run the T5 encoder and return per-token hidden states.
    ///
    /// Shape: `(1, L + 1, d_model)` — the trailing `</s>` row is included, and
    /// there is no BOS row.
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        let ids = self.encode(sequence)?;
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("T5 model mutex was poisoned by an earlier panic"))?;
        Ok(model.forward(&ids)?)
    }

    fn model_name(&self) -> &str {
        self.name
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant's fallback config must be *constructible* and agree with
    /// its registry card.
    ///
    /// This is not a formality. `fallback_config` is built eagerly by
    /// `model_info`, so a panic in it takes down loading before the hub is ever
    /// contacted — which is exactly what happened while adding Ankh: the gated
    /// activation was first built by deserializing the string `"gated-gelu"`,
    /// but candle maps that name through a *field-level* `deserialize_with` on
    /// `Config`, so `ActivationWithOptionalGating`'s own derived `Deserialize`
    /// rejected it. Nothing without a 2.9 GB download caught that.
    #[test]
    fn test_fallback_configs_are_constructible_and_match_their_cards() {
        for model in [
            T5Models::ProtT5XlHalfUniref50Enc,
            T5Models::AnkhBase,
            T5Models::AnkhLarge,
        ] {
            let card = model.card();
            let config = model.fallback_config();
            let id = model.registry_id();
            assert_eq!(config.d_model, card.metadata.d_model, "{id}: d_model");
            assert_eq!(
                config.num_layers, card.metadata.n_layers,
                "{id}: num_layers"
            );
            assert_eq!(
                config.vocab_size, card.metadata.vocab_size,
                "{id}: vocab_size"
            );
            assert!(config.num_heads > 0, "{id}: num_heads must be set");
            assert!(config.d_ff > 0, "{id}: d_ff must be set");
            assert!(
                !config.use_cache,
                "{id}: use_cache must be forced off — see T5Runner"
            );
        }
    }

    /// Ankh is gated, ProtT5 is not. Getting this backwards loads a
    /// `T5DenseGatedActDense` against plain-dense weights (or the reverse),
    /// which fails on tensor shapes rather than silently — but only after a
    /// multi-gigabyte download.
    #[test]
    fn test_only_ankh_is_gated() {
        assert!(
            !T5Models::ProtT5XlHalfUniref50Enc
                .fallback_config()
                .feed_forward_proj
                .gated,
            "ProtT5 is feed_forward_proj: \"relu\", not gated"
        );
        for model in [T5Models::AnkhBase, T5Models::AnkhLarge] {
            let ff = model.fallback_config().feed_forward_proj;
            assert!(ff.gated, "{}: Ankh is gated-gelu", model.registry_id());
            assert_eq!(
                ff.activation,
                candle_nn::Activation::NewGelu,
                "{}: gated-gelu means the tanh-approximation GELU",
                model.registry_id()
            );
        }
    }

    /// `Rostlab/ProstT5_fp16` ships `"tie_word_embeddings": null`, which
    /// candle's `#[serde(default)]` does *not* absorb — serde's `default`
    /// covers a missing key, not a present-but-null one.
    ///
    /// Without the null-stripping in [`parse_t5_config`] this config is
    /// rejected wholesale and the loader silently falls back to a built-in one.
    /// For ProstT5 that flips `tie_word_embeddings`, which decides whether
    /// `decode` scales by `sqrt(d_model)` and projects through the shared
    /// embedding — and that checkpoint has no `lm_head` to fall back on.
    #[test]
    fn test_null_valued_config_keys_are_treated_as_absent() {
        let json = r#"{
            "vocab_size": 150, "d_model": 1024, "d_kv": 128, "d_ff": 16384,
            "num_layers": 24, "num_decoder_layers": 24, "num_heads": 32,
            "relative_attention_num_buckets": 32, "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-06, "initializer_factor": 1.0,
            "feed_forward_proj": "relu", "is_encoder_decoder": true,
            "tie_word_embeddings": null,
            "pad_token_id": 0, "eos_token_id": 1, "decoder_start_token_id": 0
        }"#;
        assert!(
            serde_json::from_str::<T5Config>(json).is_err(),
            "candle rejects an explicit null — if this ever starts passing, \
             the strip in parse_t5_config is no longer needed"
        );
        let config = parse_t5_config(json).expect("nulls should be stripped");
        assert_eq!(config.d_model, 1024);
        assert!(
            config.tie_word_embeddings,
            "with the key treated as absent, candle's default (true) applies — \
             which is what HuggingFace means by writing null"
        );
    }

    /// The published precisions differ, and defaulting either to the other
    /// wastes memory or loses fidelity (ferritin-100.9).
    #[test]
    fn test_published_dtypes() {
        assert_eq!(
            T5Models::ProtT5XlHalfUniref50Enc.published_dtype(),
            DType::F16
        );
        assert_eq!(T5Models::AnkhBase.published_dtype(), DType::F32);
        assert_eq!(T5Models::AnkhLarge.published_dtype(), DType::F32);
    }
}
