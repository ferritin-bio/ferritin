//! AbLang2-architecture parity against the official Python implementation
//! (ferritin-sxg, ferritin-4dm).
//!
//! Two checkpoints share this architecture and run the same checks, in the
//! `ablang2` and `tcrlang` modules below:
//!
//! | checkpoint | fixture | inputs |
//! |---|---|---|
//! | `ablang2-paired` | `ablang2_parity` | AbLang2 README VH/VL |
//! | `tcrlang-paired` | `tcrlang_parity` | JM22 TCR, TRB first, TRA second |
//!
//! Each fixture is produced by `scripts/generate_ablang2_fixtures.py --model
//! <id>` from `oxpig/AbLang2` at a pinned commit and that checkpoint's Zenodo
//! archive, pinned by checksum. Both are committed, and every test loads its
//! fixture with [`ParityFixture`], which **errors** when it is missing rather
//! than skipping.
//!
//! Two tiers:
//!
//! - Tokenizer tests need only the fixture and run on every `cargo test`.
//! - Model tests need the 166 MB Zenodo archive, so they are `#[ignore]`d and
//!   run in the nightly (`--include-ignored`). Set `FERRITIN_ABLANG2_DIR` or
//!   `FERRITIN_TCRLANG_DIR` to an unpacked weights archive to use a local copy
//!   instead of the download; `AbLang2Runner::from_dir` checks its SHA-256
//!   either way.
//!
//! # Tolerances
//!
//! CPU F32 against PyTorch 2.15 CPU F32. Measured worst cases for
//! `ablang2-paired` (2026-11-03); `tcrlang-paired` is within the same ranges
//! (logits 3.8e-5, hidden 7.5e-6, layer 12 2.0e-5, scores 4e-7; 2026-11-05):
//!
//! | quantity | worst abs. error | bound |
//! |---|---|---|
//! | masked-LM logits, six single inputs | 3.4e-5 | 1e-3 |
//! | logits, padded batch of three | 4.5e-5 | 1e-3 |
//! | final hidden states | 3.9e-6 | 1e-3 |
//! | layer 0 / 1 / 12 (pre final LayerNorm) | 0 / 1.5e-5 / 4.9e-4 | cosine ≥ 0.99999 |
//! | pseudo-log-likelihood, confidence | 3e-7 | 1e-4 |
//!
//! The pre-final-LayerNorm residual stream is unnormalised and grows with
//! depth, so the intermediates are compared by cosine, which is scale-free.
//! Each bound sits one to two orders of magnitude above the measured error.
//! The defects this port is most exposed to — a dropped bias, swapped SwiGLU
//! halves, half-split instead of interleaved rotary, a single rather than
//! double attention scale — change the computation itself, not the rounding,
//! so they are not expected to hide inside bounds this tight.

mod support;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use ferritin_plms::ablang2::tokenizer::{self, Chain, TokenRole};
use ferritin_plms::loader::LoadOptions;
use ferritin_plms::plm_runner::PlmRunner;
use ferritin_plms::{AbLang2ChainRunner, AbLang2Models, AbLang2Runner};
use support::parity::{ParityFixture, assert_embeddings_close, assert_logits_close};

/// Max |Δ| on masked-LM logits. Measured worst case 4.5e-5 (module docs).
const LOGIT_TOL: f32 = 1e-3;
/// Max |Δ| on hidden states (post-LayerNorm, so O(1) values).
const HIDDEN_TOL: f32 = 1e-3;
/// Per-row cosine floor on hidden states.
const COSINE_FLOOR: f32 = 0.99999;
/// |Δ| on the scalar scores (mean log-probabilities).
const SCORE_TOL: f32 = 1e-4;

// The inputs, transcribed from the generator. A transcription slip shows up
// immediately as an id mismatch in `tokenizer_matches_upstream_ids`.

/// The VH/VL pair from the AbLang2 README.
const VH: &str = "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS";
const VL: &str = "DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK";

/// The JM22 TCR from upstream's TCRLang notebook: beta first, alpha second.
const TRB: &str = "GGITQSPKYLFRKEGQNVTLSCEQNLNHDAMYWYRQDPGQGLRLIYYSQIVNDFQKGDIAEGYSVSREKKESFPLTVTSAQKNPTAFYLCASSIRSSYEQYFGPGTRLTVTEDLKN";
const TRA: &str = "QLLEQSPQFLSIQEGENLTVYCNSSSVFSSLQWYRQEPGEGPVLLVTVVTGGEVKKLKRLTFQFGDARKDSSLHITAAQPGDTGLYLCAGAGSQGNLIFGKGTKLSVKP";

fn masked(seq: &str, positions: &[usize]) -> String {
    seq.chars()
        .enumerate()
        .map(|(i, c)| if positions.contains(&i) { '*' } else { c })
        .collect()
}

/// One checkpoint's fixture and inputs. Both checkpoints run every check.
struct Spec {
    model: AbLang2Models,
    fixture: &'static str,
    /// Env var naming a local unpacked archive, instead of downloading.
    dir_env: &'static str,
    first: &'static str,
    second: &'static str,
    first_only: &'static str,
    second_only: &'static str,
    first_masks: &'static [usize],
    second_masks: &'static [usize],
}

const ABLANG2: Spec = Spec {
    model: AbLang2Models::Paired,
    fixture: "ablang2_parity",
    dir_env: "FERRITIN_ABLANG2_DIR",
    first: VH,
    second: VL,
    first_only: "heavy_only",
    second_only: "light_only",
    // CDR-H3 at 99..101.
    first_masks: &[0, 30, 99, 100, 101],
    second_masks: &[5, 90],
};

const TCRLANG: Spec = Spec {
    model: AbLang2Models::TcrLangPaired,
    fixture: "tcrlang_parity",
    dir_env: "FERRITIN_TCRLANG_DIR",
    first: TRB,
    second: TRA,
    first_only: "beta_only",
    second_only: "alpha_only",
    // Inside CDR3-beta, which starts at 90.
    first_masks: &[0, 30, 95, 96, 97],
    second_masks: &[5, 90],
};

impl Spec {
    fn cases(&self) -> Vec<(&'static str, String, String)> {
        let (a, b) = (self.first, self.second);
        vec![
            ("paired", a.into(), b.into()),
            (self.first_only, a.into(), String::new()),
            (self.second_only, String::new(), b.into()),
            ("short_pair", a[..30].into(), b[..25].into()),
            (
                "masked",
                masked(a, self.first_masks),
                masked(b, self.second_masks),
            ),
            (
                "unknown",
                format!("{}X{}", &a[..20], &a[21..40]),
                format!("X{}", &b[1..30]),
            ),
        ]
    }

    fn case(&self, name: &str) -> (String, String) {
        let (_, h, l) = self
            .cases()
            .into_iter()
            .find(|(n, _, _)| *n == name)
            .expect("known case");
        (h, l)
    }

    fn batch(&self) -> [&'static str; 3] {
        ["short_pair", self.first_only, "unknown"]
    }

    fn fixture(&self) -> Result<ParityFixture> {
        ParityFixture::load_with_generator_args(
            self.fixture,
            "ablang2",
            &format!("--model {}", self.model.registry_id()),
            &Device::Cpu,
        )
    }

    fn runner(&self) -> Result<AbLang2Runner> {
        let opts = LoadOptions::new(Device::Cpu);
        match std::env::var_os(self.dir_env) {
            Some(dir) => AbLang2Runner::from_dir(self.model, std::path::Path::new(&dir), &opts),
            None => AbLang2Runner::from_pretrained_with(self.model, &opts),
        }
    }
}

fn ids_of(fx: &ParityFixture, key: &str) -> Result<Vec<u32>> {
    Ok(fx
        .tensor(key)?
        .to_dtype(DType::I64)?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|v| v as u32)
        .collect())
}

/// Largest |a - b| over two same-shaped tensors, for reporting.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?)
}

// ── Tokenizer: fixture only ──────────────────────────────────────────────────

fn check_tokenizer_matches_upstream_ids(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    for (name, h, l) in spec.cases() {
        let enc = tokenizer::encode_pair(&h, &l)?;
        let expected = ids_of(&fx, &format!("ids.{name}"))?;
        assert_eq!(enc.ids, expected, "{name}: token ids differ from upstream");
    }
    Ok(())
}

fn check_batch_ids_are_right_padded_with_pad(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    let batch = fx.tensor("batch.ids")?.to_dtype(DType::I64)?;
    let (rows, width) = batch.dims2()?;
    assert_eq!(rows, spec.batch().len());
    for (row, name) in spec.batch().iter().enumerate() {
        let (h, l) = spec.case(name);
        let enc = tokenizer::encode_pair(&h, &l)?;
        let got: Vec<u32> = batch
            .i(row)?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        assert_eq!(&got[..enc.len()], enc.ids.as_slice(), "{name}");
        assert!(
            got[enc.len()..].iter().all(|&t| t == tokenizer::PAD_ID),
            "{name}: upstream pads with '-' ({})",
            tokenizer::PAD_ID
        );
        assert!(enc.len() <= width);
    }
    Ok(())
}

/// The roles are this port's own claim, so they are checked against the
/// upstream text form independently of the ids.
fn check_roles_follow_upstream_text_form(spec: &Spec) -> Result<()> {
    for (name, h, l) in spec.cases() {
        let enc = tokenizer::encode_pair(&h, &l)?;
        let text = tokenizer::decode(&enc.ids)?;
        let expected = format!("<{h}>|<{l}>").replace("<>", "");
        assert_eq!(text, expected, "{name}");
        for (row, role) in enc.roles.iter().enumerate() {
            let c = text.as_bytes()[row] as char;
            match role {
                TokenRole::ChainStart(_) => assert_eq!(c, '<', "{name} row {row}"),
                TokenRole::ChainEnd(_) => assert_eq!(c, '>', "{name} row {row}"),
                TokenRole::Separator => assert_eq!(c, '|', "{name} row {row}"),
                TokenRole::Residue { chain, index } => {
                    let src = match chain {
                        Chain::Heavy => &h,
                        Chain::Light => &l,
                    };
                    assert_eq!(src.as_bytes()[*index] as char, c, "{name} row {row}");
                }
            }
        }
        assert_eq!(enc.residue_rows(Chain::Heavy).len(), h.len(), "{name}");
        assert_eq!(enc.residue_rows(Chain::Light).len(), l.len(), "{name}");
    }
    Ok(())
}

// ── Model: fixture + weights ─────────────────────────────────────────────────

fn check_tied_head_matches_embedding(spec: &Spec) -> Result<()> {
    let r = spec.runner()?;
    let d = max_abs_diff(r.model().embedding_weights(), r.model().head_weights())?;
    assert_eq!(d, 0.0, "AbHead.weights should be the tied embedding matrix");
    Ok(())
}

fn check_logits_match_upstream_for_every_case(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    let r = spec.runner()?;
    for (name, h, l) in spec.cases() {
        let (logits, _) = r.logits_pair(&h, &l)?;
        let reference = fx.tensor(&format!("logits.{name}"))?;
        println!(
            "{name}: max |Δ logits| = {:.3e}",
            max_abs_diff(&logits, reference)?
        );
        assert_logits_close(&logits, reference, LOGIT_TOL)
            .map_err(|e| e.context(format!("case {name}")))?;
    }
    Ok(())
}

fn check_hidden_states_match_upstream(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    let r = spec.runner()?;
    for name in ["paired", "short_pair"] {
        let (h, l) = spec.case(name);
        let (hidden, _) = r.embed_pair(&h, &l)?;
        let reference = fx.tensor(&format!("hidden.{name}"))?;
        println!(
            "{name}: max |Δ hidden| = {:.3e}",
            max_abs_diff(&hidden, reference)?
        );
        assert_logits_close(&hidden, reference, HIDDEN_TOL)?;
        assert_embeddings_close(&hidden, reference, COSINE_FLOOR)?;
    }
    Ok(())
}

/// Per-stage intermediates localise a defect to a layer when the final
/// output disagrees.
fn check_intermediate_layers_match_upstream(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    let r = spec.runner()?;
    let (h, l) = spec.case("short_pair");
    let enc = r.encode(&h, &l)?;
    let ids = Tensor::new(enc.ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let out = r.forward_ids(&ids, &[0, 1, 12])?;
    for (n, t) in out.layers {
        let reference = fx.tensor(&format!("layer{n}.short_pair"))?;
        let t = t.i(0)?;
        println!("layer {n}: max |Δ| = {:.3e}", max_abs_diff(&t, reference)?);
        // The pre-LayerNorm residual stream grows with depth, so compare by
        // cosine, which is scale-free.
        assert_embeddings_close(&t, reference, COSINE_FLOOR)?;
    }
    Ok(())
}

fn check_padded_batch_matches_upstream_and_single_runs(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    let r = spec.runner()?;
    let pairs: Vec<(String, String)> = spec.batch().iter().map(|n| spec.case(n)).collect();
    let refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(h, l)| (h.as_str(), l.as_str()))
        .collect();
    let (_, logits, encs) = r.forward_batch(&refs)?;

    let reference = fx.tensor("batch.logits")?;
    assert_eq!(logits.dims(), reference.dims());
    for (row, enc) in encs.iter().enumerate() {
        // Every row, padding included: upstream computes the padded query
        // rows too, and so does this port.
        let got = logits.i(row)?;
        let want = reference.i(row)?;
        println!(
            "batch row {row}: max |Δ logits| = {:.3e}",
            max_abs_diff(&got, &want)?
        );
        assert_logits_close(&got, &want, LOGIT_TOL)?;

        // And a real row of the batch equals the same input run alone, which
        // fails if padding leaks into attention.
        let (single, _) = r.logits_pair(refs[row].0, refs[row].1)?;
        let real = got.narrow(0, 0, enc.len())?;
        let d = max_abs_diff(&real, &single)?;
        assert!(
            d < LOGIT_TOL,
            "batch row {row} differs from its single run by {d}"
        );
    }
    Ok(())
}

fn check_scores_match_upstream(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    let r = spec.runner()?;
    // PLL is one forward pass per residue, so the generator pins it on the
    // two short cases only.
    for name in ["short_pair", "unknown"] {
        let (h, l) = spec.case(name);
        let pll = r.pseudo_log_likelihood(&h, &l)?;
        let ref_pll = fx.tensor(&format!("pll.{name}"))?.to_vec1::<f32>()?[0];
        println!("{name}: pll {pll} vs {ref_pll}");
        assert!(
            (pll - ref_pll).abs() < SCORE_TOL,
            "{name}: pll {pll} vs {ref_pll}"
        );
    }
    for name in ["paired", "masked", "unknown"] {
        let (h, l) = spec.case(name);
        let conf = r.confidence(&h, &l)?;
        let ref_conf = fx.tensor(&format!("confidence.{name}"))?.to_vec1::<f32>()?[0];
        println!("{name}: confidence {conf} vs {ref_conf}");
        assert!(
            (conf - ref_conf).abs() < SCORE_TOL,
            "{name}: confidence {conf} vs {ref_conf}"
        );
    }
    Ok(())
}

fn check_restore_matches_upstream(spec: &Spec) -> Result<()> {
    let fx = spec.fixture()?;
    let r = spec.runner()?;
    let (h, l) = spec.case("masked");
    let (rh, rl) = r.restore(&h, &l)?;
    // Upstream returns the formatted text, `<H>|<L>`, stored as ASCII bytes.
    let upstream = String::from_utf8(fx.tensor("restore.masked")?.to_vec1::<u8>()?)?;
    assert_eq!(format!("<{rh}>|<{rl}>"), upstream);
    // Only masked positions may change.
    for (orig, new) in [(&h, &rh), (&l, &rl)] {
        for (a, b) in orig.chars().zip(new.chars()) {
            assert!(a == '*' || a == b);
        }
    }
    Ok(())
}

fn check_chain_runner_layouts_select_the_right_rows(spec: &Spec) -> Result<()> {
    let (h, l) = spec.case("short_pair");
    let heavy = AbLang2ChainRunner::new(spec.runner()?, Chain::Heavy);
    let light = AbLang2ChainRunner::new(spec.runner()?, Chain::Light);

    // PlmRunner's residue rows must be exactly the rows the encoding labels
    // as that chain's residues.
    for (adapter, seq, chain) in [(&heavy, &h, Chain::Heavy), (&light, &l, Chain::Light)] {
        let residues = adapter.embed_residues(seq)?.i(0)?;
        let (pair_h, pair_l) = match chain {
            Chain::Heavy => (seq.as_str(), ""),
            Chain::Light => ("", seq.as_str()),
        };
        let (hidden, enc) = adapter.inner().embed_pair(pair_h, pair_l)?;
        let rows = enc.residue_rows(chain);
        let idx = Tensor::new(
            rows.iter()
                .map(|&r| r as u32)
                .collect::<Vec<_>>()
                .as_slice(),
            &Device::Cpu,
        )?;
        assert_eq!(
            max_abs_diff(&residues, &hidden.index_select(&idx, 0)?)?,
            0.0
        );
        assert_eq!(residues.dim(0)?, seq.len());
    }
    Ok(())
}

fn check_repeated_runs_are_bitwise_identical(spec: &Spec) -> Result<()> {
    let r = spec.runner()?;
    let (h, l) = spec.case("short_pair");
    let a = r.logits_pair(&h, &l)?.0.flatten_all()?.to_vec1::<f32>()?;
    let b = r.logits_pair(&h, &l)?.0.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a, b);
    Ok(())
}

// ── The two checkpoints ─────────────────────────────────────────────────────

mod ablang2 {
    use super::*;

    #[test]
    fn tokenizer_matches_upstream_ids() -> Result<()> {
        check_tokenizer_matches_upstream_ids(&ABLANG2)
    }

    #[test]
    fn batch_ids_are_right_padded_with_pad() -> Result<()> {
        check_batch_ids_are_right_padded_with_pad(&ABLANG2)
    }

    #[test]
    fn roles_follow_upstream_text_form() -> Result<()> {
        check_roles_follow_upstream_text_form(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn tied_head_matches_embedding() -> Result<()> {
        check_tied_head_matches_embedding(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn logits_match_upstream_for_every_case() -> Result<()> {
        check_logits_match_upstream_for_every_case(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn hidden_states_match_upstream() -> Result<()> {
        check_hidden_states_match_upstream(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn intermediate_layers_match_upstream() -> Result<()> {
        check_intermediate_layers_match_upstream(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn padded_batch_matches_upstream_and_single_runs() -> Result<()> {
        check_padded_batch_matches_upstream_and_single_runs(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn scores_match_upstream() -> Result<()> {
        check_scores_match_upstream(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn restore_matches_upstream() -> Result<()> {
        check_restore_matches_upstream(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn chain_runner_layouts_select_the_right_rows() -> Result<()> {
        check_chain_runner_layouts_select_the_right_rows(&ABLANG2)
    }

    #[test]
    #[ignore = "downloads the ablang2 weights from Zenodo (~166 MB)"]
    fn repeated_runs_are_bitwise_identical() -> Result<()> {
        check_repeated_runs_are_bitwise_identical(&ABLANG2)
    }
}

mod tcrlang {
    use super::*;

    #[test]
    fn tokenizer_matches_upstream_ids() -> Result<()> {
        check_tokenizer_matches_upstream_ids(&TCRLANG)
    }

    #[test]
    fn batch_ids_are_right_padded_with_pad() -> Result<()> {
        check_batch_ids_are_right_padded_with_pad(&TCRLANG)
    }

    #[test]
    fn roles_follow_upstream_text_form() -> Result<()> {
        check_roles_follow_upstream_text_form(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn tied_head_matches_embedding() -> Result<()> {
        check_tied_head_matches_embedding(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn logits_match_upstream_for_every_case() -> Result<()> {
        check_logits_match_upstream_for_every_case(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn hidden_states_match_upstream() -> Result<()> {
        check_hidden_states_match_upstream(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn intermediate_layers_match_upstream() -> Result<()> {
        check_intermediate_layers_match_upstream(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn padded_batch_matches_upstream_and_single_runs() -> Result<()> {
        check_padded_batch_matches_upstream_and_single_runs(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn scores_match_upstream() -> Result<()> {
        check_scores_match_upstream(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn restore_matches_upstream() -> Result<()> {
        check_restore_matches_upstream(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn chain_runner_layouts_select_the_right_rows() -> Result<()> {
        check_chain_runner_layouts_select_the_right_rows(&TCRLANG)
    }

    #[test]
    #[ignore = "downloads the tcrlang weights from Zenodo (~166 MB)"]
    fn repeated_runs_are_bitwise_identical() -> Result<()> {
        check_repeated_runs_are_bitwise_identical(&TCRLANG)
    }
}
