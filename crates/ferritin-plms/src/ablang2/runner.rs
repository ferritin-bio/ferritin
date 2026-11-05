//! Loading AbLang2 and running it on heavy/light pairs.
//!
//! [`AbLang2Runner`] is the native interface: it takes a `(heavy, light)` pair,
//! returns token-row outputs together with the [`AbLang2Encoding`] that says
//! what each row is, and implements upstream's scoring modes —
//! `pseudo_log_likelihood`, `confidence` and `restore`.
//!
//! [`AbLang2ChainRunner`] adapts it to [`PlmRunner`] for one *named* chain. The
//! trait takes a single `&str`, and for a paired model a bare string is
//! ambiguous: upstream encodes a heavy-only input as `<H>|` and a light-only
//! one as `|<L>`, and the model was trained to tell them apart. So the adapter
//! is constructed with the chain fixed, and its [`SpecialTokenLayout`] follows
//! from that choice rather than from a guess.

use super::model::{AbLang2, AbLang2Config, AbLang2Output};
use super::tokenizer::{
    self, AMINO_ACIDS, AbLang2Encoding, Chain, MASK_ID, NON_SCORABLE_IDS, PAD_ID,
};
use crate::loader::{Format, LoadOptions, WeightSource};
use crate::plm_runner::{ModelMetadata, PlmRunner, SpecialTokenLayout, pad_token_batch};
use crate::registry::{self, ModelCard};
use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use std::path::Path;

/// SHA-256 of `model.pt` inside the AbLang2 Zenodo archive.
///
/// Checked by [`AbLang2Runner::from_dir`] when the caller supplies their own
/// copy, because that path does not go through the archive hash.
pub const MODEL_PT_SHA256: &str =
    "56d6f07862a6f824f88c8707bbc03e4026c9db762be2d3041e9767e2e6f86386";

/// SHA-256 of `ablang2-weights.tar.gz` on Zenodo record 10185169.
///
/// Zenodo publishes the MD5 (`6425b35e9b83750fde67a6a6d240d996`); the SHA-256
/// was computed from the archive whose MD5 matched it.
pub const ARCHIVE_SHA256: &str = "7acb23b880e3b69b3fea8b7d6cacac0fda147daafd99edc411e0ff5c61252ac5";

/// The official AbLang2 weights: `ablang2-weights.tar.gz` on Zenodo.
pub const ZENODO_PAIRED: WeightSource = WeightSource::zenodo_archive(
    "zenodo/10185169",
    "ablang2-weights.tar.gz",
    ARCHIVE_SHA256,
    Format::Pth { root_key: None },
);

/// SHA-256 of `model.pt` inside the TCRLang Zenodo archive.
pub const TCRLANG_MODEL_PT_SHA256: &str =
    "5f93dcf715271b8315967c930ed858b890476bdc7ecb1cb08f959b900299c83b";

/// SHA-256 of `tcrlang-weights.tar.gz` on Zenodo record 11208211.
///
/// Zenodo publishes the MD5 (`8bbade5ba653096a490467cbc68b4034`); the SHA-256
/// was computed from the archive whose MD5 matched it.
pub const TCRLANG_ARCHIVE_SHA256: &str =
    "027e32a05805df1a2a46f1ec378b80c02c8bac0f0cdd272cf102a8f14f2f1626";

/// The official TCRLang-paired weights: `tcrlang-weights.tar.gz` on Zenodo.
pub const ZENODO_TCRLANG: WeightSource = WeightSource::zenodo_archive(
    "zenodo/11208211",
    "tcrlang-weights.tar.gz",
    TCRLANG_ARCHIVE_SHA256,
    Format::Pth { root_key: None },
);

/// How many masked copies of a sequence go through the model at once when
/// computing a pseudo-log-likelihood. Upstream sends them all at once; at
/// ~230 tokens that is a 230-row batch, which is fine on a workstation but
/// needlessly peaky, and chunking does not change the result.
const PLL_CHUNK: usize = 32;

/// Available checkpoints of the AbLang2 architecture.
///
/// Both take two chains in fixed slots. The API names them heavy and light
/// after AbLang2; for TCRLang the first slot is the **beta** chain (TRB) and
/// the second the **alpha** chain (TRA), the order upstream's TCRLang
/// notebook prescribes. Swapping them still runs, and gives wrong answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbLang2Models {
    /// `ablang2-paired`, trained on paired heavy/light antibody variable
    /// domains.
    Paired,
    /// `tcrlang-paired`, the same architecture trained on paired TCR
    /// variable domains (Observed TCR Space). Weights CC-BY-4.0.
    TcrLangPaired,
}

impl AbLang2Models {
    /// This variant's registry id.
    pub const fn registry_id(&self) -> &'static str {
        match self {
            Self::Paired => "ablang2-paired",
            Self::TcrLangPaired => "tcrlang-paired",
        }
    }

    /// SHA-256 of this checkpoint's `model.pt`.
    pub const fn model_pt_sha256(&self) -> &'static str {
        match self {
            Self::Paired => MODEL_PT_SHA256,
            Self::TcrLangPaired => TCRLANG_MODEL_PT_SHA256,
        }
    }

    /// This variant's [`ModelCard`].
    pub fn card(&self) -> &'static ModelCard {
        registry::lookup(self.registry_id())
            .expect("every AbLang2Models variant must have a registry entry")
    }
}

/// AbLang2 loaded and ready to run.
pub struct AbLang2Runner {
    model: AbLang2,
    device: Device,
    /// Registry id of the loaded checkpoint, or `"ablang2-custom"` for one
    /// loaded through [`from_var_builder`][Self::from_var_builder].
    id: &'static str,
}

impl AbLang2Runner {
    /// Download (once) and load the official weights at F32.
    pub fn from_pretrained(model: AbLang2Models, device: Device) -> Result<Self> {
        Self::from_pretrained_with(model, &LoadOptions::new(device))
    }

    /// Download (once) and load with an explicit device and dtype.
    ///
    /// The archive is fetched from Zenodo into ferritin's cache
    /// (`FERRITIN_CACHE`, else `~/.cache/ferritin`) and its SHA-256 checked
    /// before it is unpacked.
    pub fn from_pretrained_with(model: AbLang2Models, opts: &LoadOptions) -> Result<Self> {
        let card = model.card();
        let config_path = card.source.fetch("hparams.json")?;
        let config = AbLang2Config::from_file(&config_path)?;
        let vb = card.source.var_builder(card.file, opts)?;
        Ok(Self::from_var_builder(vb, &config, opts)?.named(model.registry_id()))
    }

    /// Load from a directory holding an unpacked weights archive
    /// (`model.pt` and `hparams.json`) for `model`.
    ///
    /// `model.pt` is checked against [`AbLang2Models::model_pt_sha256`]
    /// first: a directory is whatever the caller says it is, and the two
    /// checkpoints have identical shapes, so loading TCRLang as AbLang2 (or
    /// the reverse) would otherwise succeed silently.
    pub fn from_dir(model: AbLang2Models, dir: &Path, opts: &LoadOptions) -> Result<Self> {
        let weights = dir.join("model.pt");
        let digest = crate::loader::sha256_file(&weights)?;
        let expected = model.model_pt_sha256();
        if digest != expected {
            bail!(
                "{} has sha256 {digest}, not the official {} checkpoint ({expected})",
                weights.display(),
                model.registry_id()
            );
        }
        let config = AbLang2Config::from_file(&dir.join("hparams.json"))?;
        let vb =
            crate::loader::var_builder_from_path(&weights, Format::Pth { root_key: None }, opts)?;
        Ok(Self::from_var_builder(vb, &config, opts)?.named(model.registry_id()))
    }

    /// Load from any VarBuilder over the checkpoint's key layout.
    pub fn from_var_builder(
        vb: candle_nn::VarBuilder<'static>,
        config: &AbLang2Config,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let model = AbLang2::load(vb, config).context("failed to load AbLang2")?;
        Ok(Self {
            model,
            device: opts.device.clone(),
            id: "ablang2-custom",
        })
    }

    fn named(mut self, id: &'static str) -> Self {
        self.id = id;
        self
    }

    /// Registry id of the loaded checkpoint.
    pub fn id(&self) -> &'static str {
        self.id
    }

    /// The underlying model.
    pub fn model(&self) -> &AbLang2 {
        &self.model
    }

    /// Device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Tokenize a pair; see [`tokenizer::encode_pair`].
    pub fn encode(&self, heavy: &str, light: &str) -> Result<AbLang2Encoding> {
        tokenizer::encode_pair(heavy, light)
    }

    fn ids_tensor(&self, enc: &AbLang2Encoding) -> Result<Tensor> {
        Ok(Tensor::new(enc.ids.as_slice(), &self.device)?.unsqueeze(0)?)
    }

    /// Encoder output for already-tokenized ids, `(batch, tokens)` U32.
    pub fn forward_ids(&self, ids: &Tensor, layers: &[usize]) -> Result<AbLang2Output> {
        self.model.encode(ids, layers)
    }

    /// Per-token hidden states `(tokens, d)` and the encoding that labels them.
    ///
    /// Special-token rows are kept, as upstream's `rescoding` keeps them; use
    /// [`AbLang2Encoding::residue_rows`] to select a chain's residues.
    pub fn embed_pair(&self, heavy: &str, light: &str) -> Result<(Tensor, AbLang2Encoding)> {
        let enc = self.encode(heavy, light)?;
        let out = self.model.encode(&self.ids_tensor(&enc)?, &[])?;
        Ok((out.hidden.i(0)?, enc))
    }

    /// Masked-LM logits `(tokens, vocab)` and the encoding that labels them —
    /// upstream's `likelihood` mode.
    pub fn logits_pair(&self, heavy: &str, light: &str) -> Result<(Tensor, AbLang2Encoding)> {
        let enc = self.encode(heavy, light)?;
        let out = self.model.encode(&self.ids_tensor(&enc)?, &[])?;
        Ok((self.model.logits(&out.hidden)?.i(0)?, enc))
    }

    /// Per-residue hidden states for one chain of a pair, `(residues, d)`.
    ///
    /// The pair is still run jointly — the light chain conditions the heavy
    /// chain's embedding and vice versa — only the rows are selected. Returns
    /// `None` when that chain is empty.
    pub fn chain_embeddings(
        &self,
        heavy: &str,
        light: &str,
        chain: Chain,
    ) -> Result<Option<Tensor>> {
        let (hidden, enc) = self.embed_pair(heavy, light)?;
        let rows = enc.residue_rows(chain);
        if rows.is_empty() {
            return Ok(None);
        }
        let idx = Tensor::new(
            rows.iter()
                .map(|&r| r as u32)
                .collect::<Vec<_>>()
                .as_slice(),
            &self.device,
        )?;
        Ok(Some(hidden.index_select(&idx, 0)?))
    }

    /// Run several pairs in one padded forward pass.
    ///
    /// Returns `(hidden, logits, encodings)` with hidden `(B, T_max, d)` and
    /// logits `(B, T_max, vocab)`. Rows past each encoding's length are padding
    /// and carry whatever the model computes there (as upstream's do); trust
    /// only the first `encodings[i].len()` rows of row `i`.
    pub fn forward_batch(
        &self,
        pairs: &[(&str, &str)],
    ) -> Result<(Tensor, Tensor, Vec<AbLang2Encoding>)> {
        let encs = pairs
            .iter()
            .map(|(h, l)| self.encode(h, l))
            .collect::<Result<Vec<_>>>()?;
        let rows: Vec<Vec<u32>> = encs.iter().map(|e| e.ids.clone()).collect();
        let batch = pad_token_batch(&rows, PAD_ID, &self.device)?;
        let out = self.model.encode(&batch.ids, &[])?;
        let logits = self.model.logits(&out.hidden)?;
        Ok((out.hidden, logits, encs))
    }

    /// Mean log-probability of each scorable residue given everything else,
    /// with that residue masked — upstream's `pseudo_log_likelihood`.
    ///
    /// One forward pass per scorable residue. Positions holding `*` or `X`
    /// are neither masked nor scored, and the special-token logit columns are
    /// excluded from the softmax, exactly as upstream does. Higher (closer to
    /// zero) is more antibody-like.
    pub fn pseudo_log_likelihood(&self, heavy: &str, light: &str) -> Result<f32> {
        let enc = self.encode(heavy, light)?;
        let positions = enc.scorable_rows();
        if positions.is_empty() {
            bail!("pseudo_log_likelihood: no scorable residues (every residue is '*' or 'X')");
        }
        let len = enc.len();
        let mut total = 0f64;
        for chunk in positions.chunks(PLL_CHUNK) {
            let mut ids = Vec::with_capacity(chunk.len() * len);
            for &pos in chunk {
                let mut row = enc.ids.clone();
                row[pos] = MASK_ID;
                ids.extend(row);
            }
            let ids = Tensor::from_vec(ids, (chunk.len(), len), &self.device)?;
            let out = self.model.encode(&ids, &[])?;
            let log_probs = residue_log_probs(&self.model.logits(&out.hidden)?)?;
            for (row, &pos) in chunk.iter().enumerate() {
                let label = enc.ids[pos] as usize - 1;
                total += log_probs.i((row, pos, label))?.to_scalar::<f32>()? as f64;
            }
        }
        Ok((total / positions.len() as f64) as f32)
    }

    /// Mean log-probability of each scorable residue from one *unmasked*
    /// forward pass — upstream's `confidence`.
    pub fn confidence(&self, heavy: &str, light: &str) -> Result<f32> {
        let enc = self.encode(heavy, light)?;
        let positions = enc.scorable_rows();
        if positions.is_empty() {
            bail!("confidence: no scorable residues (every residue is '*' or 'X')");
        }
        let out = self.model.encode(&self.ids_tensor(&enc)?, &[])?;
        let log_probs = residue_log_probs(&self.model.logits(&out.hidden)?)?.i(0)?;
        let mut total = 0f64;
        for &pos in &positions {
            let label = enc.ids[pos] as usize - 1;
            total += log_probs.i((pos, label))?.to_scalar::<f32>()? as f64;
        }
        Ok((total / positions.len() as f64) as f32)
    }

    /// Fill every `*` with the most likely amino acid — upstream's `restore`
    /// without alignment.
    ///
    /// Returns the restored `(heavy, light)`. Only positions that were `*`
    /// change; the argmax is over the twenty amino acids, never a special
    /// token.
    pub fn restore(&self, heavy: &str, light: &str) -> Result<(String, String)> {
        let (logits, enc) = self.logits_pair(heavy, light)?;
        let best = logits
            .narrow(D::Minus1, 1, AMINO_ACIDS.len())?
            .argmax(D::Minus1)?
            .to_vec1::<u32>()?;
        let mut restored = [heavy.chars().collect::<Vec<_>>(), light.chars().collect()];
        for (row, role) in enc.roles.iter().enumerate() {
            if let tokenizer::TokenRole::Residue { chain, index } = *role
                && enc.ids[row] == MASK_ID
            {
                let slot = match chain {
                    Chain::Heavy => 0,
                    Chain::Light => 1,
                };
                restored[slot][index] = AMINO_ACIDS[best[row] as usize];
            }
        }
        let [h, l] = restored;
        Ok((h.into_iter().collect(), l.into_iter().collect()))
    }

    /// Architecture dimensions.
    pub fn metadata(&self) -> ModelMetadata {
        let cfg = self.model.config();
        ModelMetadata {
            d_model: cfg.hidden_embed_size,
            n_layers: cfg.n_encoder_blocks,
            vocab_size: cfg.vocab_size,
            // Rotary positions only, and no length limit in upstream code.
            max_positions: None,
        }
    }
}

/// Log-softmax over the twenty amino-acid columns only, `(…, 20)` in F32.
///
/// Upstream sets the special-token columns to `-inf` and then takes a full
/// cross-entropy, which is the same thing: those columns contribute nothing
/// to the normaliser. Column `j` here is token id `j + 1`.
fn residue_log_probs(logits: &Tensor) -> Result<Tensor> {
    debug_assert!(NON_SCORABLE_IDS.iter().all(|&id| id == 0 || id > 20));
    let aa = logits
        .narrow(D::Minus1, 1, AMINO_ACIDS.len())?
        .to_dtype(DType::F32)?;
    Ok(candle_nn::ops::log_softmax(&aa, D::Minus1)?)
}

/// [`AbLang2Runner`] presented as a [`PlmRunner`] for one named chain.
///
/// `embed(seq)` runs `seq` as that chain alone. The layout is fixed by the
/// choice: heavy is `<H…>|` (one leading, two trailing special rows), light
/// is `|<L…>` (two leading, one trailing).
pub struct AbLang2ChainRunner {
    runner: AbLang2Runner,
    chain: Chain,
    name: String,
}

impl AbLang2ChainRunner {
    /// Wrap a runner, fixing which chain single sequences are.
    pub fn new(runner: AbLang2Runner, chain: Chain) -> Self {
        let slot = match chain {
            Chain::Heavy => "first chain",
            Chain::Light => "second chain",
        };
        let name = format!("{} ({slot})", runner.id());
        Self {
            runner,
            chain,
            name,
        }
    }

    /// The wrapped runner, for paired inputs and scoring.
    pub fn inner(&self) -> &AbLang2Runner {
        &self.runner
    }

    fn pair<'a>(&self, sequence: &'a str) -> (&'a str, &'a str) {
        match self.chain {
            Chain::Heavy => (sequence, ""),
            Chain::Light => ("", sequence),
        }
    }
}

impl PlmRunner for AbLang2ChainRunner {
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        let (h, l) = self.pair(sequence);
        Ok(self.runner.embed_pair(h, l)?.0.unsqueeze(0)?)
    }

    fn logits(&self, sequence: &str) -> Result<Tensor> {
        let (h, l) = self.pair(sequence);
        Ok(self.runner.logits_pair(h, l)?.0.unsqueeze(0)?)
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn special_tokens(&self) -> SpecialTokenLayout {
        match self.chain {
            Chain::Heavy => SpecialTokenLayout {
                leading: 1,
                trailing: 2,
            },
            Chain::Light => SpecialTokenLayout {
                leading: 2,
                trailing: 1,
            },
        }
    }

    fn metadata(&self) -> ModelMetadata {
        self.runner.metadata()
    }

    fn device(&self) -> &Device {
        self.runner.device()
    }
}
