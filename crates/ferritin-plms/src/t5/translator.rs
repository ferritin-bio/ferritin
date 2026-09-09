//! ProstT5 — translation between the amino-acid and 3Di structural alphabets
//! (ferritin-goh.6).
//!
//! ProstT5 is the only encoder-**decoder** in the crate, and the only model
//! whose useful output is generated tokens rather than an embedding. It
//! therefore sits outside [`PlmRunner`][crate::plm_runner::PlmRunner], which is
//! about embeddings, rather than being contorted to fit it.
//!
//! What it buys: the 3Di string it produces is what SaProt consumes, so this
//! pair removes the Foldseek dependency from the structure-aware path.

use crate::loader::{LoadOptions, WeightSource};
use crate::registry::{self, ModelCard};
use crate::t5::runner::parse_t5_config;
use crate::t5::tokenizer::{self, Direction};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5::{Config as T5Config, T5ForConditionalGeneration};
use std::sync::Mutex;

/// Available ProstT5 variants.
pub enum ProstT5Models {
    /// `Rostlab/ProstT5_fp16` — the half-precision build. The F32
    /// `Rostlab/ProstT5` is the same weights at 11.3 GB instead of 5.6 GB.
    XlFp16,
}

impl ProstT5Models {
    /// This variant's registry id.
    pub const fn registry_id(&self) -> &'static str {
        match self {
            Self::XlFp16 => "prostt5-fp16",
        }
    }

    /// This variant's [`ModelCard`].
    pub fn card(&self) -> &'static ModelCard {
        registry::lookup(self.registry_id())
            .expect("every ProstT5Models variant must have a registry entry")
    }

    /// The precision this checkpoint is published at.
    pub const fn published_dtype(&self) -> DType {
        DType::F16
    }

    /// The published `config.json`, transcribed, for when the hub is
    /// unreachable.
    ///
    /// Note `tie_word_embeddings: true`. The checkpoint has **no `lm_head`
    /// tensor**, so the output projection is the shared embedding transposed
    /// and the hidden state is scaled by `sqrt(d_model)` first.
    ///
    /// Setting it to `false` would fail loudly — candle would go looking for
    /// an `lm_head` that is not there. The quiet failure is upstream of that:
    /// the hub's `config.json` writes `"tie_word_embeddings": null`, and if
    /// that config were dropped wholesale (see `parse_t5_config`) this fallback
    /// is what would be used instead. So it has to be right.
    fn fallback_config(&self) -> T5Config {
        T5Config {
            vocab_size: 150,
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
            tie_word_embeddings: true,
            is_decoder: false,
            is_encoder_decoder: true,
            // The decoder genuinely caches; see `ProstT5Translator`.
            use_cache: true,
            pad_token_id: tokenizer::PAD_ID as usize,
            eos_token_id: tokenizer::EOS_ID as usize,
            decoder_start_token_id: Some(tokenizer::PAD_ID as usize),
        }
    }
}

/// ProstT5, wrapped for greedy translation.
///
/// # Why the model sits behind a `Mutex`, and why it matters more here
///
/// [`T5Runner`][crate::t5::runner::T5Runner] also holds its model in a `Mutex`,
/// but there it guards nothing: `T5Attention::load` sets
/// `use_cache: cfg.use_cache && decoder`, and an encoder stack loads with
/// `decoder = false`, so the encoder never caches.
///
/// **The decoder does.** Every `decode` step appends to a per-layer KV cache,
/// which is what makes incremental generation fast — and what makes a stale
/// cache dangerous: the next translation would concatenate its keys onto the
/// previous one's and produce a fluent, wrong answer with no error. So
/// [`translate`][Self::translate] clears the cache before it starts, and
/// `test_repeated_translation_is_deterministic` fails if that is ever removed.
pub struct ProstT5Translator {
    model: Mutex<T5ForConditionalGeneration>,
    device: Device,
}

impl ProstT5Translator {
    /// Load from the hub at the checkpoint's published precision (F16).
    pub fn from_pretrained(model: ProstT5Models, device: Device) -> Result<Self> {
        let dtype = model.published_dtype();
        Self::from_pretrained_with(model, &LoadOptions::new(device).with_dtype(dtype))
    }

    /// Load with an explicit device and dtype.
    pub fn from_pretrained_with(model: ProstT5Models, opts: &LoadOptions) -> Result<Self> {
        let card = model.card();
        let config = load_config(&card.source, model.fallback_config());
        let vb = card.source.var_builder(card.file, opts)?;
        let inner = T5ForConditionalGeneration::load(vb, &config).with_context(|| {
            format!(
                "failed to load {} as a T5 encoder-decoder",
                card.source.repo_id
            )
        })?;
        Ok(Self {
            model: Mutex::new(inner),
            device: opts.device.clone(),
        })
    }

    /// Translate `sequence` between the amino-acid and 3Di alphabets.
    ///
    /// Greedy decoding, which is what ProstT5's own usage does: the mapping is
    /// close to deterministic and sampling would only add noise to a structural
    /// annotation.
    ///
    /// ProstT5's translation is **length-preserving**: one output token per
    /// input residue. Length, not `</s>`, is therefore the stopping criterion,
    /// and the EOS logit is suppressed at every step — which is exactly what
    /// Rostlab's own inference does by passing `min_length == max_length` to
    /// `generate`, and what HuggingFace implements as `MinLengthLogitsProcessor`.
    ///
    /// Breaking on `</s>` instead would silently return a truncated structure
    /// for any sequence where the model wanted to stop early, and would
    /// disagree with the reference on exactly those cases.
    pub fn translate(&self, sequence: &str, direction: Direction) -> Result<String> {
        let residues = tokenizer::residue_count(sequence);
        if residues == 0 {
            bail!("ProstT5: cannot translate an empty sequence");
        }
        let ids = tokenizer::encode_for_translation(sequence, direction);
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;

        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("ProstT5 model mutex was poisoned by an earlier panic"))?;

        // Mandatory, not hygiene: a cache left over from the previous call is
        // concatenated onto, not overwritten.
        model.clear_kv_cache();
        let encoded = model.encode(&input)?;

        let mut generated: Vec<u32> = Vec::with_capacity(residues);
        // T5 starts the decoder from `decoder_start_token_id`, which is <pad>.
        let mut next_input = vec![tokenizer::PAD_ID];
        for _ in 0..residues {
            let decoder_input = Tensor::new(next_input.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = model
                .decode(&decoder_input, &encoded)?
                .to_dtype(DType::F32)?;
            let next = argmax_excluding_eos(&logits)?;
            generated.push(next);
            // Only the new token: the KV cache holds the rest.
            next_input = vec![next];
        }
        drop(model);

        decode_generated(&generated, direction)
    }
}

/// Greedy pick, with `</s>` masked out.
///
/// The mask is why this is not a plain `argmax`: generation runs for a fixed
/// number of steps, so an EOS that wins the argmax must be passed over in
/// favour of the next-best real token rather than ending the sequence. Setting
/// it to `NEG_INFINITY` is safe here because the tensor has already been cast
/// to F32 — at F16 an infinity would be a live value in later arithmetic, but
/// nothing follows this except the argmax itself.
fn argmax_excluding_eos(logits: &Tensor) -> Result<u32> {
    let mut row: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let eos = tokenizer::EOS_ID as usize;
    if eos < row.len() {
        row[eos] = f32::NEG_INFINITY;
    }
    let (best, _) =
        row.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
    Ok(best as u32)
}

/// Turn generated ids back into a string in the direction's output alphabet.
///
/// Errors on a token outside that alphabet rather than substituting a
/// placeholder: a `?` in a 3Di string would flow onward into SaProt as if it
/// were a structural state.
fn decode_generated(ids: &[u32], direction: Direction) -> Result<String> {
    ids.iter()
        .enumerate()
        .map(|(i, &id)| {
            let decoded = match direction {
                // AA in, 3Di out — and vice versa.
                Direction::AaToFold => tokenizer::three_di_char(id),
                Direction::FoldToAa => tokenizer::residue_char(id),
            };
            decoded.ok_or_else(|| {
                anyhow::anyhow!(
                    "ProstT5 generated token {id} at position {i}, which is not in the \
                     {} alphabet this direction produces",
                    match direction {
                        Direction::AaToFold => "3Di",
                        Direction::FoldToAa => "amino-acid",
                    }
                )
            })
        })
        .collect()
}

/// Load `config.json` from the hub, falling back to the built-in config.
///
/// Unlike the encoder-only path this does **not** force `use_cache` off — the
/// decoder's cache is what makes incremental generation possible.
fn load_config(source: &WeightSource, fallback: T5Config) -> T5Config {
    match source.fetch_optional("config.json") {
        None => {
            eprintln!(
                "warning: {}: could not download config.json; using the built-in config.",
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
                     using the built-in config instead.",
                    source.repo_id
                );
                fallback
            }
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_config_matches_the_card() {
        let model = ProstT5Models::XlFp16;
        let card = model.card();
        let config = model.fallback_config();
        assert_eq!(config.d_model, card.metadata.d_model);
        assert_eq!(config.num_layers, card.metadata.n_layers);
        assert_eq!(config.vocab_size, card.metadata.vocab_size);
        assert!(
            config.tie_word_embeddings,
            "ProstT5 has no lm_head tensor; the head is the shared embedding"
        );
        assert!(
            config.use_cache,
            "the decoder's KV cache is what makes incremental generation work"
        );
    }

    fn logits_row(values: &[(u32, f32)]) -> Tensor {
        let mut row = vec![-10.0f32; 150];
        for &(id, v) in values {
            row[id as usize] = v;
        }
        Tensor::from_vec(row, (1, 150), &Device::Cpu).unwrap()
    }

    /// `</s>` is never selected, even when it wins outright.
    ///
    /// This is the whole reason generation does not use a plain `argmax`:
    /// ProstT5's translation is length-preserving, so an EOS that tops the
    /// distribution has to be passed over for the next-best real token — the
    /// same thing HuggingFace's `MinLengthLogitsProcessor` does when Rostlab's
    /// inference passes `min_length == max_length`.
    #[test]
    fn test_eos_is_never_greedily_selected() {
        let runner_up = tokenizer::three_di_id('v');
        let logits = logits_row(&[(tokenizer::EOS_ID, 99.0), (runner_up, 5.0)]);
        assert_eq!(
            argmax_excluding_eos(&logits).unwrap(),
            runner_up,
            "EOS should be masked out in favour of the next-best token"
        );
    }

    /// Masking EOS must not disturb the ordinary case.
    #[test]
    fn test_argmax_is_otherwise_ordinary() {
        let best = tokenizer::three_di_id('d');
        let logits = logits_row(&[(best, 7.0), (tokenizer::three_di_id('v'), 6.5)]);
        assert_eq!(argmax_excluding_eos(&logits).unwrap(), best);
    }

    /// Decoding reads the *output* alphabet, which is the opposite of the
    /// input's — the direction names which way the translation runs, not which
    /// alphabet comes back.
    #[test]
    fn test_decode_generated_uses_the_output_alphabet() {
        let three_di: Vec<u32> = "dvq".chars().map(tokenizer::three_di_id).collect();
        assert_eq!(
            decode_generated(&three_di, Direction::AaToFold).unwrap(),
            "dvq"
        );

        let residues: Vec<u32> = "MKT".chars().map(tokenizer::residue_id).collect();
        assert_eq!(
            decode_generated(&residues, Direction::FoldToAa).unwrap(),
            "MKT"
        );
    }

    /// A token from the wrong alphabet is an error, not a placeholder: a `?`
    /// in a 3Di string would flow onward into SaProt as a structural state.
    #[test]
    fn test_decode_generated_rejects_out_of_alphabet_tokens() {
        let err = decode_generated(&[tokenizer::residue_id('M')], Direction::AaToFold)
            .unwrap_err()
            .to_string();
        assert!(err.contains("3Di alphabet"), "unexpected error: {err}");
        assert!(
            decode_generated(&[tokenizer::AA_TO_FOLD_ID], Direction::AaToFold).is_err(),
            "the direction token is not a structural state"
        );
    }
}
