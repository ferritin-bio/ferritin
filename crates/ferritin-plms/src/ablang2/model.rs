//! The AbLang2 encoder and masked-LM head in Candle.
//!
//! Transcribed from `ablang2/models/ablang2/{ablang,encoderblock}.py` at the
//! pinned commit (see [`super`]). The checkpoint's own key names are used
//! unchanged — `AbRep.*` for the encoder, `AbHead.*` for the head — so a
//! missing or misnamed tensor fails at load time rather than computing
//! something else.
//!
//! Five details that each change the numbers, and are each easy to "fix" into
//! a bug when porting:
//!
//! 1. **Attention is scaled twice.** `MultiHeadAttention.forward` multiplies
//!    `q` by `head_dim ** -0.5`, and `attention` then divides the scores by
//!    `sqrt(head_dim)` again, so the effective temperature is `1 / head_dim`,
//!    not the textbook `1 / sqrt(head_dim)`. The checkpoint was trained that
//!    way, so it is reproduced, in the same order.
//! 2. **Rotary embeddings are interleaved.** `rotary-embedding-torch` rotates
//!    adjacent pairs `(x0, x1), (x2, x3), …`, which is candle's
//!    [`rope_i`][candle_nn::rotary_emb::rope_i], not the half-split
//!    [`rope`][candle_nn::rotary_emb::rope] ESM-style models use. The inverse
//!    frequencies are read from each layer's `rotary_emb.freqs` buffer rather
//!    than recomputed, so the formula cannot drift from what was trained.
//! 3. **SwiGLU puts the gate second.** `x, gate = x.chunk(2)` then
//!    `silu(gate) * x` — the *first* half is the value. Swapping the halves
//!    still produces a well-shaped tensor.
//! 4. **Every linear layer has a bias**, including all four attention
//!    projections. A `linear_no_bias` anywhere would leave a checkpoint
//!    tensor unread (ferritin's recurring dropped-bias defect class).
//! 5. **LayerNorm eps is `1e-12`**, from `hparams.json`, not PyTorch's `1e-5`.
//!
//! The masked-LM head is a `Linear → SwiGLU → LayerNorm` followed by a
//! projection onto the embedding matrix plus a bias. Upstream ties that
//! projection to `aa_embed_layer.weight`; the checkpoint stores the tied copy
//! as `AbHead.weights`, and that stored copy is what is used.

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};
use serde::Deserialize;

/// The model's `hparams.json`, as shipped in the Zenodo archive.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AbLang2Config {
    /// Checkpoint name, `"AbLang-2"`.
    #[serde(default)]
    pub name: String,
    /// Number of encoder blocks.
    pub n_encoder_blocks: usize,
    /// Hidden width.
    pub hidden_embed_size: usize,
    /// Attention heads.
    pub n_attn_heads: usize,
    /// FFN activation; only `"swiglu"` is supported (see [`AbLang2::load`]).
    pub a_fn: String,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// Padding token id.
    pub pad_tkn: u32,
    /// Start token id.
    pub start_tkn: u32,
    /// End token id.
    pub end_tkn: u32,
    /// Separator token id.
    pub sep_tkn: u32,
    /// Mask token id.
    pub mask_tkn: u32,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl AbLang2Config {
    /// The published `ablang2-paired` hyperparameters, transcribed from the
    /// archive's `hparams.json`.
    pub fn paired() -> Self {
        Self {
            name: "AbLang-2".to_string(),
            n_encoder_blocks: 12,
            hidden_embed_size: 480,
            n_attn_heads: 20,
            a_fn: "swiglu".to_string(),
            layer_norm_eps: 1e-12,
            pad_tkn: super::tokenizer::PAD_ID,
            start_tkn: super::tokenizer::START_ID,
            end_tkn: super::tokenizer::END_ID,
            sep_tkn: super::tokenizer::SEP_ID,
            mask_tkn: super::tokenizer::MASK_ID,
            vocab_size: super::tokenizer::VOCAB_SIZE,
        }
    }

    /// Read `hparams.json`.
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("{} is not an AbLang2 hparams.json", path.display()))
    }

    /// Reject configurations this port has not been checked against.
    ///
    /// The special-token ids are compiled into [`super::tokenizer`], so a
    /// checkpoint that moved them would be fed the wrong ids silently.
    pub fn validate(&self) -> Result<()> {
        use super::tokenizer as t;
        if self.a_fn != "swiglu" {
            bail!(
                "AbLang2 checkpoint uses a_fn = {:?}; only \"swiglu\" (ablang2-paired) is ported",
                self.a_fn
            );
        }
        let ids = (
            self.pad_tkn,
            self.start_tkn,
            self.end_tkn,
            self.sep_tkn,
            self.mask_tkn,
            self.vocab_size,
        );
        let expected = (
            t::PAD_ID,
            t::START_ID,
            t::END_ID,
            t::SEP_ID,
            t::MASK_ID,
            t::VOCAB_SIZE,
        );
        if ids != expected {
            bail!(
                "AbLang2 checkpoint special tokens (pad, start, end, sep, mask, vocab) = {ids:?} \
                 differ from the built-in tokenizer's {expected:?}"
            );
        }
        if !self.hidden_embed_size.is_multiple_of(self.n_attn_heads) {
            bail!(
                "hidden_embed_size {} is not divisible by n_attn_heads {}",
                self.hidden_embed_size,
                self.n_attn_heads
            );
        }
        let head_dim = self.hidden_embed_size / self.n_attn_heads;
        if !head_dim.is_multiple_of(2) {
            bail!("rotary embeddings need an even head_dim, got {head_dim}");
        }
        Ok(())
    }

    /// Per-head width.
    pub fn head_dim(&self) -> usize {
        self.hidden_embed_size / self.n_attn_heads
    }
}

/// `silu(second half) * first half`.
fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let half = xs.dim(D::Minus1)? / 2;
    let value = xs.narrow(D::Minus1, 0, half)?;
    let gate = xs.narrow(D::Minus1, half, half)?;
    Ok((candle_nn::ops::silu(&gate)? * value)?)
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    /// Rotary inverse frequencies, `(head_dim / 2,)`, F32.
    inv_freq: Tensor,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &AbLang2Config) -> Result<Self> {
        let d = cfg.hidden_embed_size;
        let head_dim = cfg.head_dim();
        // Read at F32 whatever the model dtype: these are frequencies, and
        // rounding them to F16 would move every rotary angle.
        let inv_freq = vb
            .pp("rotary_emb")
            .get_with_hints_dtype(head_dim / 2, "freqs", Default::default(), DType::F32)
            .context("rotary_emb.freqs")?;
        Ok(Self {
            q_proj: linear(d, d, vb.pp("q_proj"))?,
            k_proj: linear(d, d, vb.pp("k_proj"))?,
            v_proj: linear(d, d, vb.pp("v_proj"))?,
            out_proj: linear(d, d, vb.pp("out_proj"))?,
            inv_freq,
            n_heads: cfg.n_attn_heads,
            head_dim,
        })
    }

    /// `(cos, sin)`, each `(seq_len, head_dim / 2)` in `dtype`.
    fn rotary_tables(&self, seq_len: usize, dtype: DType) -> Result<(Tensor, Tensor)> {
        let device = self.inv_freq.device();
        let positions = Tensor::arange(0u32, seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((seq_len, 1))?;
        let freqs = positions.broadcast_mul(&self.inv_freq.reshape((1, self.head_dim / 2))?)?;
        Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
    }

    fn heads(&self, xs: Tensor, b: usize, l: usize) -> Result<Tensor> {
        Ok(xs
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    fn forward(&self, xs: &Tensor, key_mask: &Tensor) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let scaling = (self.head_dim as f64).powf(-0.5);
        // Upstream: `q *= self.scaling` before the heads are split …
        let q = (self.q_proj.forward(xs)? * scaling)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let (q, k, v) = (
            self.heads(q, b, l)?,
            self.heads(k, b, l)?,
            self.heads(v, b, l)?,
        );
        let (cos, sin) = self.rotary_tables(l, q.dtype())?;
        let q = candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?;

        // … and `attention` divides by sqrt(head_dim) again. See module docs.
        let scores = (q.matmul(&k.t()?.contiguous()?)? / (self.head_dim as f64).sqrt())?;
        let scores = scores.broadcast_add(key_mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = weights.matmul(&v)?.transpose(1, 2)?.reshape((b, l, d))?;
        Ok(self.out_proj.forward(&out)?)
    }
}

struct EncoderBlock {
    attention: Attention,
    pre_attn_layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
    ffn_in: Linear,
    ffn_out: Linear,
}

impl EncoderBlock {
    fn load(vb: VarBuilder, cfg: &AbLang2Config) -> Result<Self> {
        let d = cfg.hidden_embed_size;
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            attention: Attention::load(vb.pp("multihead_attention"), cfg)?,
            pre_attn_layer_norm: layer_norm(d, eps, vb.pp("pre_attn_layer_norm"))?,
            final_layer_norm: layer_norm(d, eps, vb.pp("final_layer_norm"))?,
            // `intermediate_layer` is Sequential(Linear(d, 4d * 2), SwiGLU,
            // Linear(4d, d)); index 1 is the parameter-free activation.
            ffn_in: linear(d, d * 4 * 2, vb.pp("intermediate_layer.0"))?,
            ffn_out: linear(d * 4, d, vb.pp("intermediate_layer.2"))?,
        })
    }

    fn forward(&self, xs: &Tensor, key_mask: &Tensor) -> Result<Tensor> {
        let attn = self
            .attention
            .forward(&self.pre_attn_layer_norm.forward(xs)?, key_mask)?;
        let xs = (xs + attn)?;
        let ffn = self.ffn_in.forward(&self.final_layer_norm.forward(&xs)?)?;
        let ffn = self.ffn_out.forward(&swiglu(&ffn)?)?;
        Ok((xs + ffn)?)
    }
}

struct Head {
    dense: Linear,
    norm: LayerNorm,
    /// `(vocab, d)` — the stored copy of the tied embedding matrix.
    weights: Tensor,
    bias: Tensor,
}

impl Head {
    fn load(vb: VarBuilder, cfg: &AbLang2Config) -> Result<Self> {
        let d = cfg.hidden_embed_size;
        Ok(Self {
            // `ff` is Sequential(Linear(d, 2d), SwiGLU, LayerNorm(d)).
            dense: linear(d, d * 2, vb.pp("ff.0"))?,
            norm: layer_norm(d, cfg.layer_norm_eps, vb.pp("ff.2"))?,
            weights: vb.get((cfg.vocab_size, d), "weights")?,
            bias: vb.get(cfg.vocab_size, "bias")?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let xs = self.norm.forward(&swiglu(&self.dense.forward(hidden)?)?)?;
        let (b, l, d) = xs.dims3()?;
        let logits = xs.reshape((b * l, d))?.matmul(&self.weights.t()?)?;
        Ok(logits
            .broadcast_add(&self.bias)?
            .reshape((b, l, self.weights.dim(0)?))?)
    }
}

/// Encoder output for one batch.
#[derive(Debug, Clone)]
pub struct AbLang2Output {
    /// `(batch, tokens, d)` after the final LayerNorm — upstream's
    /// `last_hidden_states`, i.e. what `rescoding` returns.
    pub hidden: Tensor,
    /// Requested intermediate states, `(layer, (batch, tokens, d))`, in the
    /// order requested. Layer 0 is the embedding output and layer `n` the
    /// output of block `n`, both *before* the final LayerNorm — upstream's
    /// `return_rep_layers`.
    pub layers: Vec<(usize, Tensor)>,
}

/// The AbLang2 encoder plus its masked-LM head.
pub struct AbLang2 {
    embed: Embedding,
    blocks: Vec<EncoderBlock>,
    final_norm: LayerNorm,
    head: Head,
    config: AbLang2Config,
}

impl AbLang2 {
    /// Load from a VarBuilder rooted at the checkpoint's top level.
    pub fn load(vb: VarBuilder, config: &AbLang2Config) -> Result<Self> {
        config.validate()?;
        let rep = vb.pp("AbRep");
        let d = config.hidden_embed_size;
        let blocks = (0..config.n_encoder_blocks)
            .map(|i| {
                EncoderBlock::load(rep.pp(format!("encoder_blocks.{i}")), config)
                    .with_context(|| format!("AbLang2 encoder block {i}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed: embedding(config.vocab_size, d, rep.pp("aa_embed_layer"))?,
            blocks,
            final_norm: layer_norm(
                d,
                config.layer_norm_eps,
                rep.pp("layer_norm_after_encoder_blocks"),
            )?,
            head: Head::load(vb.pp("AbHead"), config).context("AbLang2 head")?,
            config: config.clone(),
        })
    }

    /// The configuration this model was loaded with.
    pub fn config(&self) -> &AbLang2Config {
        &self.config
    }

    /// The input embedding matrix, `(vocab, d)`.
    pub fn embedding_weights(&self) -> &Tensor {
        self.embed.embeddings()
    }

    /// The head's stored projection matrix, `(vocab, d)`.
    pub fn head_weights(&self) -> &Tensor {
        &self.head.weights
    }

    /// `(batch, 1, 1, tokens)`: `0` for real keys, `-inf` for padding —
    /// upstream's `masked_fill(tokens == pad, -inf)` on the key axis.
    fn key_mask(&self, ids: &Tensor, dtype: DType) -> Result<Tensor> {
        let (b, l) = ids.dims2()?;
        let is_pad = ids.eq(self.config.pad_tkn)?;
        let zeros = Tensor::zeros((b, l), dtype, ids.device())?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, (b, l), ids.device())?.to_dtype(dtype)?;
        Ok(is_pad.where_cond(&neg_inf, &zeros)?.reshape((b, 1, 1, l))?)
    }

    /// Run the encoder over `(batch, tokens)` ids, padded with
    /// [`PAD_ID`][super::tokenizer::PAD_ID].
    ///
    /// `layers` lists the intermediate states to keep (see
    /// [`AbLang2Output::layers`]).
    pub fn encode(&self, ids: &Tensor, layers: &[usize]) -> Result<AbLang2Output> {
        if let Some(&bad) = layers.iter().find(|&&n| n > self.blocks.len()) {
            bail!(
                "layer {bad} requested but AbLang2 has {} blocks",
                self.blocks.len()
            );
        }
        let mut xs = self.embed.forward(ids)?;
        let key_mask = self.key_mask(ids, xs.dtype())?;
        let mut kept = Vec::new();
        let mut keep = |n: usize, t: &Tensor| {
            if layers.contains(&n) {
                kept.push((n, t.clone()));
            }
        };
        keep(0, &xs);
        for (i, block) in self.blocks.iter().enumerate() {
            xs = block.forward(&xs, &key_mask)?;
            keep(i + 1, &xs);
        }
        kept.sort_by_key(|(n, _)| layers.iter().position(|m| m == n));
        Ok(AbLang2Output {
            hidden: self.final_norm.forward(&xs)?,
            layers: kept,
        })
    }

    /// Masked-LM logits `(batch, tokens, vocab)` from encoder output.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.head.forward(hidden)
    }

    /// Device the weights live on.
    pub fn device(&self) -> &Device {
        self.embed.embeddings().device()
    }
}
