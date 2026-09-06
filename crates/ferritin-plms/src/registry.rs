//! One table describing every supported model.
//!
//! Before this, the same facts were spread across six enums that each encoded a
//! different subset in a different shape: `AmplifyModels` returned a
//! [`WeightSource`], `ESM2Models` a source plus a config, `ESMCModels` a source
//! plus a filename plus a config wrapped in `Result`, and so on. Which
//! tokenizer a model needs lived in its runner; how many special tokens it
//! wraps lived in its `PlmRunner` impl; whether it had ever been checked
//! against a Python reference lived only in the test suite.
//!
//! [`REGISTRY`] is where that belongs. A model is a data row.
//!
//! # Why a const table
//!
//! The variation between models is data — strings, dimensions, token counts —
//! so it stays data. The moment it becomes `dyn`, the compiler stops checking
//! that every model is fully specified, which is exactly the property worth
//! having: adding a row that omits a field will not compile.
//!
//! [`Family`] is a closed enum for the same reason. Adding a backbone should be
//! a deliberate, reviewed act rather than something that falls out of a string.
//!
//! ```
//! # use ferritin_plms::registry::{REGISTRY, lookup};
//! let card = lookup("esm2-t6-8m").expect("registered");
//! assert_eq!(card.metadata.d_model, 320);
//! assert_eq!(card.source.repo_id, "facebook/esm2_t6_8M_UR50D");
//! ```

use crate::loader::WeightSource;
use crate::plm_runner::{ModelMetadata, SpecialTokenLayout};

// ── Family ────────────────────────────────────────────────────────────────────

/// Architecture family a model belongs to.
///
/// Closed on purpose: a new family means a new loader, which is a reviewed
/// change rather than a new string. Families are added when a loader for them
/// lands, so every variant here has at least one row in [`REGISTRY`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// ESM-2 and its schema-compatible relatives (ESM-1v, SaProt).
    Esm2,
    /// AMPLIFY.
    Amplify,
    /// ESM Cambrian.
    Esmc,
    /// ESM3, the multi-track model.
    Esm3,
    /// ESMFold2 — structure prediction, not an embedding model.
    Esmfold2,
    /// ProteinMPNN / LigandMPNN — inverse folding, not an embedding model.
    Mpnn,
}

// ── TokenizerSpec ─────────────────────────────────────────────────────────────

/// Where a model's tokenizer comes from.
///
/// Not incidental detail: the four ported families genuinely differ here, and
/// that knowledge used to be buried in six separate runners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerSpec {
    /// A `tokenizer.json` downloaded from the model's own HF repo.
    HfJson,
    /// A `tokenizer.json` compiled into the binary with `include_bytes!`,
    /// named by its path under `src/`.
    Embedded(&'static str),
    /// A hand-written vocabulary table in Rust, named by its module path.
    BuiltinVocab(&'static str),
    /// No tokenizer: the model consumes structure, not sequence.
    None,
}

// ── ParityStatus ──────────────────────────────────────────────────────────────

/// Whether this model's numerics have been checked against a Python reference.
///
/// `Unverified` is the honest default and covers most rows. It is not a
/// statement that a model is wrong — only that nothing proves it right, which
/// is worth being able to read off the registry rather than inferring from
/// which fixtures happen to exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityStatus {
    /// A committed fixture pins this model's output against the reference
    /// implementation. Names the fixture stem under `tests/fixtures/`.
    Verified {
        /// Fixture stem, without the `.safetensors` extension.
        fixture: &'static str,
    },
    /// No parity fixture exists, so nothing checks this model's numerics.
    Unverified,
}

// ── ModelCard ─────────────────────────────────────────────────────────────────

/// Everything needed to identify, fetch, and load one model.
#[derive(Debug, Clone, Copy)]
pub struct ModelCard {
    /// Stable kebab-case identifier, unique across the registry.
    pub id: &'static str,
    /// Architecture family.
    pub family: Family,
    /// Which repo and revision the weights live in, and their on-disk format.
    pub source: WeightSource,
    /// Path of the weight file within the repo.
    pub file: &'static str,
    /// Where the tokenizer comes from.
    pub tokenizer: TokenizerSpec,
    /// How many special tokens the tokenizer wraps a sequence in.
    pub specials: SpecialTokenLayout,
    /// Architecture dimensions.
    ///
    /// For the non-embedding families these are the nearest analogue rather
    /// than a literal reading: `d_model` is the model's hidden width and
    /// `vocab_size` its output alphabet.
    pub metadata: ModelMetadata,
    /// Rough in-memory footprint at F32, from the published parameter count.
    ///
    /// Approximate on purpose — it exists to tier CI and to warn before a
    /// machine starts swapping, not to be exact. Load at F16 to roughly halve
    /// it; see `LoadOptions::with_dtype`.
    pub approx_bytes_f32: u64,
    /// Whether anything checks this model's numerics.
    pub parity: ParityStatus,
    /// Set when the model cannot currently be loaded, saying why.
    ///
    /// Three ported models are in this state, each for a recorded reason. A
    /// row that carries this is present because the model is part of the
    /// public API, not because it works.
    pub unsupported: Option<&'static str>,
}

impl ModelCard {
    /// Whether this model can actually be loaded today.
    pub const fn is_loadable(&self) -> bool {
        self.unsupported.is_none()
    }

    /// Whether this model consumes sequence and can implement
    /// [`PlmRunner`][crate::plm_runner::PlmRunner].
    ///
    /// A property of the card, not of its [`Family`]: ESM3 contains both a
    /// sequence model and a VQ-VAE structure encoder that takes backbone
    /// coordinates, so the family alone does not settle it.
    pub const fn is_embedding_model(&self) -> bool {
        !matches!(self.tokenizer, TokenizerSpec::None)
    }

    /// Lowercase family name, for matching against string-keyed tables.
    pub const fn family_str(&self) -> &'static str {
        match self.family {
            Family::Esm2 => "esm2",
            Family::Amplify => "amplify",
            Family::Esmc => "esmc",
            Family::Esm3 => "esm3",
            Family::Esmfold2 => "esmfold2",
            Family::Mpnn => "proteinmpnn",
        }
    }
}

const GB: u64 = 1024 * 1024 * 1024;
const MB: u64 = 1024 * 1024;

/// Every model the public API exposes.
///
/// Rows carrying [`unsupported`][ModelCard::unsupported] are still listed: they
/// are reachable from the public API, and a registry that hid them would
/// misrepresent what a caller can name.
pub const REGISTRY: &[ModelCard] = &[
    // ── ESM-2 ────────────────────────────────────────────────────────────────
    ModelCard {
        id: "esm2-t6-8m",
        family: Family::Esm2,
        source: WeightSource::safetensors("facebook/esm2_t6_8M_UR50D").at_revision("main"),
        file: "model.safetensors",
        tokenizer: TokenizerSpec::Embedded("esm2/tokenizer.json"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 320,
            n_layers: 6,
            vocab_size: 33,
            max_positions: Some(1026),
        },
        approx_bytes_f32: 32 * MB,
        parity: ParityStatus::Verified {
            fixture: "esm2_parity",
        },
        unsupported: None,
    },
    ModelCard {
        id: "esm2-t12-35m",
        family: Family::Esm2,
        source: WeightSource::safetensors("facebook/esm2_t12_35M_UR50D").at_revision("main"),
        file: "model.safetensors",
        tokenizer: TokenizerSpec::Embedded("esm2/tokenizer.json"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 480,
            n_layers: 12,
            vocab_size: 33,
            max_positions: Some(1026),
        },
        approx_bytes_f32: 140 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    ModelCard {
        id: "esm2-t30-150m",
        family: Family::Esm2,
        source: WeightSource::safetensors("facebook/esm2_t30_150M_UR50D").at_revision("main"),
        file: "model.safetensors",
        tokenizer: TokenizerSpec::Embedded("esm2/tokenizer.json"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 640,
            n_layers: 30,
            vocab_size: 33,
            max_positions: Some(1026),
        },
        approx_bytes_f32: 600 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    ModelCard {
        id: "esm2-t33-650m",
        family: Family::Esm2,
        source: WeightSource::safetensors("facebook/esm2_t33_650M_UR50D").at_revision("main"),
        file: "model.safetensors",
        tokenizer: TokenizerSpec::Embedded("esm2/tokenizer.json"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 1280,
            n_layers: 33,
            vocab_size: 33,
            max_positions: Some(1026),
        },
        approx_bytes_f32: 2 * GB + 600 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    ModelCard {
        id: "esm2-t36-3b",
        family: Family::Esm2,
        source: WeightSource::safetensors("facebook/esm2_t36_3B_UR50D").at_revision("main"),
        file: "model.safetensors",
        tokenizer: TokenizerSpec::Embedded("esm2/tokenizer.json"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 2560,
            n_layers: 36,
            vocab_size: 33,
            max_positions: Some(1026),
        },
        approx_bytes_f32: 12 * GB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    ModelCard {
        id: "esm2-t48-15b",
        family: Family::Esm2,
        source: WeightSource::safetensors("facebook/esm2_t48_15B_UR50D").at_revision("main"),
        file: "model.safetensors",
        tokenizer: TokenizerSpec::Embedded("esm2/tokenizer.json"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 5120,
            n_layers: 48,
            vocab_size: 33,
            max_positions: Some(1026),
        },
        approx_bytes_f32: 60 * GB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    // ── AMPLIFY ──────────────────────────────────────────────────────────────
    ModelCard {
        id: "amplify-120m",
        family: Family::Amplify,
        source: WeightSource::safetensors("chandar-lab/AMPLIFY_120M").at_revision("main"),
        file: "model.safetensors",
        // The runner downloads the repo's tokenizer.json; an embedded copy also
        // exists for AMPLIFY::load_tokenizer.
        tokenizer: TokenizerSpec::HfJson,
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 640,
            n_layers: 24,
            vocab_size: 27,
            max_positions: Some(2048),
        },
        approx_bytes_f32: 480 * MB,
        parity: ParityStatus::Verified {
            fixture: "amplify_parity",
        },
        unsupported: None,
    },
    ModelCard {
        id: "amplify-350m",
        family: Family::Amplify,
        source: WeightSource::safetensors("chandar-lab/AMPLIFY_350M").at_revision("main"),
        file: "model.safetensors",
        tokenizer: TokenizerSpec::HfJson,
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 960,
            n_layers: 32,
            vocab_size: 27,
            max_positions: Some(2048),
        },
        approx_bytes_f32: GB + 400 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    // ── ESM Cambrian ─────────────────────────────────────────────────────────
    ModelCard {
        id: "esmc-300m",
        family: Family::Esmc,
        source: WeightSource::pth("EvolutionaryScale/esmc-300m-2024-12", None),
        file: "data/weights/esmc_300m_2024_12_v0.pth",
        tokenizer: TokenizerSpec::BuiltinVocab("esmc::tokenizer::EsmSequenceTokenizer"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 960,
            n_layers: 30,
            vocab_size: 64,
            // Rotary positions: no hard architectural cap.
            max_positions: None,
        },
        approx_bytes_f32: GB + 200 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    ModelCard {
        id: "esmc-600m",
        family: Family::Esmc,
        source: WeightSource::pth("EvolutionaryScale/esmc-600m-2024-12", None),
        file: "data/weights/esmc_600m_2024_12_v0.pth",
        tokenizer: TokenizerSpec::BuiltinVocab("esmc::tokenizer::EsmSequenceTokenizer"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 1152,
            n_layers: 36,
            vocab_size: 64,
            max_positions: None,
        },
        approx_bytes_f32: 2 * GB + 400 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    ModelCard {
        id: "esmc-6b",
        family: Family::Esmc,
        source: WeightSource::safetensors("EvolutionaryScale/esmc-6b-2024-12"),
        // Sharded across six files; the loader follows this index
        // (ferritin-100.24).
        file: "model.safetensors.index.json",
        tokenizer: TokenizerSpec::BuiltinVocab("esmc::tokenizer::EsmSequenceTokenizer"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 2560,
            n_layers: 80,
            vocab_size: 64,
            max_positions: None,
        },
        approx_bytes_f32: 24 * GB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    // ── ESM3 ─────────────────────────────────────────────────────────────────
    ModelCard {
        id: "esm3-sm-open-v1",
        family: Family::Esm3,
        source: WeightSource::pth("EvolutionaryScale/esm3-sm-open-v1", None),
        file: "data/weights/esm3_sm_open_v1.pth",
        tokenizer: TokenizerSpec::BuiltinVocab("esm3::tokenization::sequence"),
        specials: SpecialTokenLayout::BOS_EOS,
        metadata: ModelMetadata {
            d_model: 1536,
            n_layers: 48,
            vocab_size: 64,
            max_positions: None,
        },
        approx_bytes_f32: 5 * GB + 600 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
    ModelCard {
        id: "esm3-structure-encoder-v0",
        family: Family::Esm3,
        source: WeightSource::pth("EvolutionaryScale/esm3-sm-open-v1", None),
        file: "data/weights/esm3_structure_encoder_v0.pth",
        // Consumes backbone coordinates and emits structure tokens.
        tokenizer: TokenizerSpec::None,
        specials: SpecialTokenLayout::NONE,
        metadata: ModelMetadata {
            d_model: 1024,
            n_layers: 2,
            // Codebook size: the structure-token alphabet it emits.
            vocab_size: 4096,
            max_positions: None,
        },
        approx_bytes_f32: 30 * MB,
        parity: ParityStatus::Unverified,
        unsupported: Some(
            "the ported VQ-VAE encoder is a different shape from the released checkpoint: its \
             blocks have no multi-head attn, it roots at 'transformer' not 'encoder', and it \
             carries a relative positional embedding this port does not model (ferritin-100.22)",
        ),
    },
    // ── ESMFold2 ─────────────────────────────────────────────────────────────
    ModelCard {
        id: "esmfold2-fast",
        family: Family::Esmfold2,
        source: WeightSource::safetensors("biohub/ESMFold2-Fast"),
        file: "model.safetensors",
        // Consumes ESMC-6B hidden states, not tokens of its own.
        tokenizer: TokenizerSpec::None,
        specials: SpecialTokenLayout::NONE,
        metadata: ModelMetadata {
            // d_single, the single-representation width.
            d_model: 384,
            n_layers: 24,
            // Structure output, no token vocabulary.
            vocab_size: 0,
            max_positions: None,
        },
        approx_bytes_f32: 755 * MB,
        parity: ParityStatus::Unverified,
        unsupported: Some(
            "the ported architecture does not match the released checkpoint: none of its 1032 \
             tensors resolve to a model parameter (ferritin-100.17)",
        ),
    },
    // ── ProteinMPNN ──────────────────────────────────────────────────────────
    ModelCard {
        id: "proteinmpnn-v48-020",
        family: Family::Mpnn,
        source: WeightSource::pth("zcpbx/ligandmpnn-weights", Some("model_state_dict"))
            .at_revision("main"),
        file: "model_params/proteinmpnn_v_48_020.pt",
        // Consumes structure; residues are encoded by the featurizer.
        tokenizer: TokenizerSpec::None,
        specials: SpecialTokenLayout::NONE,
        metadata: ModelMetadata {
            d_model: 128,
            // 3 encoder + 3 decoder.
            n_layers: 6,
            // 20 amino acids plus X.
            vocab_size: 21,
            max_positions: None,
        },
        approx_bytes_f32: 7 * MB,
        parity: ParityStatus::Unverified,
        unsupported: None,
    },
];

// ── Support matrix ────────────────────────────────────────────────────────────

/// Render [`REGISTRY`] as a markdown support matrix.
///
/// The column that matters is **Parity**. "Does it compile" and even "does it
/// load" are not what a user needs to know before trusting a number — what
/// they need is whether anyone has ever compared this port's output against
/// the reference implementation. Today two models have, and the table says so
/// rather than leaving it to be inferred from which fixtures happen to exist
/// (ferritin-100.13).
///
/// The copy embedded in the crate docs is checked against this function by
/// `test_lib_rs_support_matrix_is_current`, so the two cannot drift.
pub fn support_matrix_markdown() -> String {
    let mut out = String::new();
    out.push_str("| Model | Family | Weights | Parity | Status |\n");
    out.push_str("|---|---|---|---|---|\n");

    for card in REGISTRY {
        let format = match card.source.format {
            crate::loader::Format::Safetensors => "safetensors",
            crate::loader::Format::Pth { .. } => "pth",
        };
        let parity = match card.parity {
            ParityStatus::Verified { fixture } => {
                format!("verified (`{fixture}`)")
            }
            ParityStatus::Unverified => "**not checked**".to_string(),
        };
        let status = match card.unsupported {
            None => "supported".to_string(),
            Some(reason) => {
                // Keep the table readable; the full reason lives on the card.
                let short = reason.split(" (ferritin-").next().unwrap_or(reason);
                let issue = reason
                    .rsplit_once("(ferritin-")
                    .map(|(_, tail)| tail.trim_end_matches(')'))
                    .unwrap_or("");
                let first = short.split(&[',', ':'][..]).next().unwrap_or(short);
                if issue.is_empty() {
                    format!("**unsupported** — {first}")
                } else {
                    format!("**unsupported** — {first} (ferritin-{issue})")
                }
            }
        };
        out.push_str(&format!(
            "| `{}` | {:?} | `{}` ({format}) | {parity} | {status} |\n",
            card.id, card.family, card.source.repo_id,
        ));
    }
    out
}

// ── Lookups ───────────────────────────────────────────────────────────────────

/// Find a model by its registry id.
pub fn lookup(id: &str) -> Option<&'static ModelCard> {
    REGISTRY.iter().find(|c| c.id == id)
}

/// Every card in a family.
pub fn by_family(family: Family) -> impl Iterator<Item = &'static ModelCard> {
    REGISTRY.iter().filter(move |c| c.family == family)
}

/// Every card that can actually be loaded today.
pub fn loadable() -> impl Iterator<Item = &'static ModelCard> {
    REGISTRY.iter().filter(|c| c.is_loadable())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_ids_are_unique() {
        let mut seen = HashSet::new();
        for card in REGISTRY {
            assert!(seen.insert(card.id), "duplicate registry id: {}", card.id);
        }
    }

    /// Ids are the registry's public handle, so they stay kebab-case and free
    /// of the underscores and capitals the upstream repo names use.
    #[test]
    fn test_ids_are_kebab_case() {
        for card in REGISTRY {
            assert!(
                card.id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "id {:?} should be lowercase kebab-case",
                card.id
            );
        }
    }

    /// Every repo id must be well formed, or the failure only shows up as a
    /// confusing 404 after a network round-trip.
    #[test]
    fn test_every_source_repo_is_well_formed() {
        for card in REGISTRY {
            assert!(
                card.source.repo_id.split('/').count() == 2
                    && !card.source.repo_id.starts_with('/')
                    && !card.source.repo_id.ends_with('/'),
                "{}: malformed repo id {:?}",
                card.id,
                card.source.repo_id
            );
            assert!(!card.file.is_empty(), "{}: empty weight filename", card.id);
        }
    }

    #[test]
    fn test_metadata_dimensions_are_plausible() {
        for card in REGISTRY {
            assert!(card.metadata.d_model > 0, "{}: zero d_model", card.id);
            assert!(card.metadata.n_layers > 0, "{}: zero n_layers", card.id);
            assert!(
                card.approx_bytes_f32 > 0,
                "{}: zero approx_bytes_f32",
                card.id
            );
        }
    }

    /// Embedding models tokenize and therefore have a vocabulary; the
    /// structure models deliberately have neither.
    #[test]
    fn test_embedding_models_have_a_tokenizer_and_vocab() {
        for card in REGISTRY {
            if card.is_embedding_model() {
                assert_ne!(
                    card.tokenizer,
                    TokenizerSpec::None,
                    "{}: an embedding model needs a tokenizer",
                    card.id
                );
                assert!(
                    card.metadata.vocab_size > 0,
                    "{}: an embedding model needs a vocabulary",
                    card.id
                );
            } else {
                assert_eq!(
                    card.metadata.max_positions, None,
                    "{}: a structure model has no token positions to cap",
                    card.id
                );
            }
        }
    }

    /// Every family variant must have at least one row. Family is closed so
    /// that adding a backbone is deliberate; a variant with no models means
    /// the enum has drifted ahead of the loaders.
    #[test]
    fn test_every_family_has_at_least_one_model() {
        for family in [
            Family::Esm2,
            Family::Amplify,
            Family::Esmc,
            Family::Esm3,
            Family::Esmfold2,
            Family::Mpnn,
        ] {
            assert!(
                by_family(family).next().is_some(),
                "{family:?} has no models; drop the variant or add its loader"
            );
        }
    }

    /// The three unsupported models are the ones with recorded reasons. This
    /// pins the set so that a model silently becoming unloadable — or quietly
    /// staying that way after a fix — shows up here.
    #[test]
    fn test_unsupported_models_are_the_known_set() {
        let mut unsupported: Vec<&str> = REGISTRY
            .iter()
            .filter(|c| !c.is_loadable())
            .map(|c| c.id)
            .collect();
        unsupported.sort_unstable();
        assert_eq!(unsupported, ["esm3-structure-encoder-v0", "esmfold2-fast"]);

        for card in REGISTRY.iter().filter(|c| !c.is_loadable()) {
            let reason = card.unsupported.unwrap();
            assert!(
                reason.contains("ferritin-"),
                "{}: an unsupported reason should cite its tracking issue; got: {reason}",
                card.id
            );
        }
    }

    /// Parity claims must name a fixture that the test suite actually has.
    /// Only ESM2-8M and AMPLIFY-120M are verified today.
    #[test]
    fn test_verified_models_name_a_real_fixture() {
        let mut verified: Vec<(&str, &str)> = REGISTRY
            .iter()
            .filter_map(|c| match c.parity {
                ParityStatus::Verified { fixture } => Some((c.id, fixture)),
                ParityStatus::Unverified => None,
            })
            .collect();
        verified.sort_unstable();
        assert_eq!(
            verified,
            [
                ("amplify-120m", "amplify_parity"),
                ("esm2-t6-8m", "esm2_parity"),
            ],
            "the set of parity-verified models changed; that is a deliberate act"
        );
    }

    #[test]
    fn test_lookup_finds_and_misses() {
        assert_eq!(lookup("esm2-t6-8m").map(|c| c.id), Some("esm2-t6-8m"));
        assert!(lookup("no-such-model").is_none());
    }

    #[test]
    fn test_loadable_excludes_unsupported() {
        assert!(loadable().all(|c| c.unsupported.is_none()));
        assert!(!loadable().any(|c| c.id == "esmfold2-fast"));
    }
}

#[cfg(test)]
mod matrix {
    use super::*;

    /// Regeneration helper: prints the matrix for pasting into `lib.rs`.
    ///
    /// ```shell
    /// cargo test -p ferritin-plms --lib print_support_matrix -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "prints the matrix for copying into lib.rs"]
    fn print_support_matrix() {
        println!("{}", support_matrix_markdown());
    }

    /// The copy in the crate docs must match what the registry renders.
    ///
    /// A stale matrix is worse than none: it would tell a user a model is
    /// parity-verified, or supported, after that stopped being true.
    #[test]
    fn test_lib_rs_support_matrix_is_current() {
        const LIB_RS: &str = include_str!("lib.rs");
        const BEGIN: &str = "//! <!-- BEGIN SUPPORT MATRIX -->";
        const END: &str = "//! <!-- END SUPPORT MATRIX -->";

        let start = LIB_RS
            .find(BEGIN)
            .expect("lib.rs should carry a BEGIN SUPPORT MATRIX marker")
            + BEGIN.len();
        let end = LIB_RS
            .find(END)
            .expect("lib.rs should carry an END SUPPORT MATRIX marker");

        let embedded: String = LIB_RS[start..end]
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                format!(
                    "{}\n",
                    l.trim_start().trim_start_matches("//!").trim_start()
                )
            })
            .collect();

        let rendered: String = support_matrix_markdown()
            .lines()
            .map(|l| format!("{l}\n"))
            .collect();

        assert_eq!(
            embedded, rendered,
            "the support matrix in lib.rs is stale. Regenerate it with:\n  \
             cargo test -p ferritin-plms --lib print_support_matrix -- --ignored --nocapture\n\
             then replace the block between the SUPPORT MATRIX markers."
        );
    }

    /// Every model in the matrix carries an explicit parity verdict.
    #[test]
    fn test_matrix_states_parity_for_every_model() {
        let matrix = support_matrix_markdown();
        for card in REGISTRY {
            let row = matrix
                .lines()
                .find(|l| l.contains(&format!("`{}`", card.id)))
                .unwrap_or_else(|| panic!("{} missing from the matrix", card.id));
            assert!(
                row.contains("verified") || row.contains("not checked"),
                "{}: the matrix must state a parity verdict; got: {row}",
                card.id
            );
        }
    }
}
