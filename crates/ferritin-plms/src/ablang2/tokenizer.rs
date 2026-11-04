//! AbLang2's paired heavy/light tokenizer, with an explicit token→residue map.
//!
//! Transcribed from `ablang2/models/ablang2/vocab.py` and the formatting in
//! `ablang2/pretrained.py` at the pinned commit (see [`super`]). Upstream
//! builds the string `<HEAVY>|<LIGHT>`, deletes any `<>` left by an empty
//! chain, and maps each character through a 26-entry table. So:
//!
//! | input | text | tokens |
//! |---|---|---|
//! | heavy + light | `<H…>\|<L…>` | `len(H) + len(L) + 5` |
//! | heavy only | `<H…>\|` | `len(H) + 3` |
//! | light only | `\|<L…>` | `len(L) + 3` |
//!
//! That asymmetry is why this tokenizer returns a [`TokenRole`] per token
//! instead of a fixed leading/trailing special-token count: the number of
//! special rows before the first light-chain residue depends on whether there
//! is a heavy chain, so no single [`SpecialTokenLayout`] describes a pair.
//!
//! [`SpecialTokenLayout`]: crate::plm_runner::SpecialTokenLayout
//!
//! # What is and is not residue text
//!
//! Inside a chain, upstream accepts any character in its table. Two of those
//! are meaningful residue placeholders and are accepted here too:
//!
//! - `*` — the mask token, a residue position whose identity is to be
//!   predicted (what `restore` fills in).
//! - `X` — the unknown-residue token, a residue position of unknown identity.
//!
//! The other four table entries — `<`, `>`, `|` and the padding `-` — are
//! *structure*, not residues. Upstream would tokenize them if they appeared in
//! chain text, silently changing where it thinks chains begin and end; this
//! port rejects them instead. Anything outside the table (lower case,
//! whitespace, `B`/`Z`/`U`/`O`) raises a `KeyError` upstream and an error here.

use anyhow::{Result, bail};

/// Start-of-chain token `<`.
pub const START_ID: u32 = 0;
/// Padding token `-`.
pub const PAD_ID: u32 = 21;
/// End-of-chain token `>`.
pub const END_ID: u32 = 22;
/// Mask token `*`.
pub const MASK_ID: u32 = 23;
/// Unknown-residue token `X`.
pub const UNKNOWN_ID: u32 = 24;
/// Heavy/light separator token `|`.
pub const SEP_ID: u32 = 25;
/// Vocabulary size, including the specials.
pub const VOCAB_SIZE: usize = 26;

/// The twenty amino acids in id order: `M` = 1 … `L` = 20.
///
/// Upstream's order, not alphabetical. Ids 1..=20 are exactly the range
/// `restore` takes its argmax over.
pub const AMINO_ACIDS: [char; 20] = [
    'M', 'R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'F', 'Y', 'W',
    'L',
];

/// The ids upstream's scoring treats as "not a scorable residue":
/// `tokenizer.all_special_tokens` — pad, start, end, separator, mask and
/// unknown. Their logit columns are set to `-inf` before scoring, and
/// positions holding them are never scored.
pub const NON_SCORABLE_IDS: [u32; 6] = [PAD_ID, START_ID, END_ID, SEP_ID, MASK_ID, UNKNOWN_ID];

/// Which antibody chain a residue belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Chain {
    /// The heavy chain (VH), always first.
    Heavy,
    /// The light chain (VL), always second.
    Light,
}

/// What one token stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenRole {
    /// `<` opening a chain.
    ChainStart(Chain),
    /// A residue position: `index` counts from 0 within its chain, so it is
    /// the position in the string the caller passed for that chain.
    Residue {
        /// The chain it belongs to.
        chain: Chain,
        /// Position within that chain's input string.
        index: usize,
    },
    /// `>` closing a chain.
    ChainEnd(Chain),
    /// `|` between the heavy and light slots. Present even for single-chain
    /// input, because upstream always emits it.
    Separator,
}

/// A tokenized heavy/light pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbLang2Encoding {
    /// Token ids, unpadded.
    pub ids: Vec<u32>,
    /// One role per id.
    pub roles: Vec<TokenRole>,
}

impl AbLang2Encoding {
    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Always false: even an empty pair is rejected before encoding.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Token rows holding residues of `chain`, in residue order.
    pub fn residue_rows(&self, chain: Chain) -> Vec<usize> {
        self.roles
            .iter()
            .enumerate()
            .filter_map(|(row, role)| match role {
                TokenRole::Residue { chain: c, .. } if *c == chain => Some(row),
                _ => None,
            })
            .collect()
    }

    /// Token rows holding residues of either chain, heavy then light.
    pub fn all_residue_rows(&self) -> Vec<usize> {
        self.roles
            .iter()
            .enumerate()
            .filter_map(|(row, role)| matches!(role, TokenRole::Residue { .. }).then_some(row))
            .collect()
    }

    /// Rows upstream would score: residues that are neither `*` nor `X`.
    pub fn scorable_rows(&self) -> Vec<usize> {
        self.ids
            .iter()
            .enumerate()
            .filter_map(|(row, id)| (!NON_SCORABLE_IDS.contains(id)).then_some(row))
            .collect()
    }
}

/// The id of a residue-text character, or `None` if it is not residue text.
///
/// Accepts the twenty amino acids, `X` and `*`; see the module docs for why
/// the other table entries are refused.
pub fn residue_id(c: char) -> Option<u32> {
    match c {
        '*' => Some(MASK_ID),
        'X' => Some(UNKNOWN_ID),
        _ => AMINO_ACIDS
            .iter()
            .position(|&aa| aa == c)
            .map(|i| i as u32 + 1),
    }
}

/// The character for an id, the inverse of upstream's table.
pub fn id_to_char(id: u32) -> Option<char> {
    match id {
        START_ID => Some('<'),
        PAD_ID => Some('-'),
        END_ID => Some('>'),
        MASK_ID => Some('*'),
        UNKNOWN_ID => Some('X'),
        SEP_ID => Some('|'),
        1..=20 => Some(AMINO_ACIDS[id as usize - 1]),
        _ => None,
    }
}

fn push_chain(enc: &mut AbLang2Encoding, chain: Chain, sequence: &str) -> Result<()> {
    if sequence.is_empty() {
        return Ok(());
    }
    enc.ids.push(START_ID);
    enc.roles.push(TokenRole::ChainStart(chain));
    for (index, c) in sequence.chars().enumerate() {
        let Some(id) = residue_id(c) else {
            let name = match chain {
                Chain::Heavy => "heavy",
                Chain::Light => "light",
            };
            bail!(
                "AbLang2 {name} chain: {c:?} at position {index} is not residue text; \
                 expected one of the 20 upper-case amino acids, 'X' (unknown) or '*' (mask)"
            );
        };
        enc.ids.push(id);
        enc.roles.push(TokenRole::Residue { chain, index });
    }
    enc.ids.push(END_ID);
    enc.roles.push(TokenRole::ChainEnd(chain));
    Ok(())
}

/// Tokenize a heavy/light pair exactly as `ablang2.pretrained` does.
///
/// Either chain may be empty, not both. Order is fixed: heavy first.
pub fn encode_pair(heavy: &str, light: &str) -> Result<AbLang2Encoding> {
    if heavy.is_empty() && light.is_empty() {
        bail!("AbLang2 needs at least one of the heavy and light chains");
    }
    let mut enc = AbLang2Encoding {
        ids: Vec::with_capacity(heavy.len() + light.len() + 5),
        roles: Vec::with_capacity(heavy.len() + light.len() + 5),
    };
    push_chain(&mut enc, Chain::Heavy, heavy)?;
    enc.ids.push(SEP_ID);
    enc.roles.push(TokenRole::Separator);
    push_chain(&mut enc, Chain::Light, light)?;
    Ok(enc)
}

/// Parse upstream's `"HEAVY|LIGHT"` string form into its two chains.
///
/// This is the form AbLang2's README tokenizes and the one its `pretrained`
/// front-end builds internally, so it is accepted as a convenience. Exactly
/// one `|` is required — `"HEAVY"` alone is ambiguous between a heavy-only and
/// a light-only input, and guessing would silently put a light chain in the
/// heavy slot.
pub fn split_pair(pair: &str) -> Result<(&str, &str)> {
    let mut parts = pair.split('|');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(h), Some(l), None) => Ok((h, l)),
        _ => bail!(
            "AbLang2 pair {pair:?} must contain exactly one '|' separating heavy from light \
             (use \"HEAVY|\" for heavy-only and \"|LIGHT\" for light-only)"
        ),
    }
}

/// Decode ids back into upstream's text form.
pub fn decode(ids: &[u32]) -> Result<String> {
    ids.iter()
        .map(|&id| id_to_char(id).ok_or_else(|| anyhow::anyhow!("{id} is not an AbLang2 token id")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_matches_upstream_table() {
        // ablang2/models/ablang2/vocab.py, verbatim.
        let upstream = [
            ('<', 0),
            ('-', 21),
            ('>', 22),
            ('*', 23),
            ('X', 24),
            ('|', 25),
            ('M', 1),
            ('R', 2),
            ('H', 3),
            ('K', 4),
            ('D', 5),
            ('E', 6),
            ('S', 7),
            ('T', 8),
            ('N', 9),
            ('Q', 10),
            ('C', 11),
            ('G', 12),
            ('P', 13),
            ('A', 14),
            ('V', 15),
            ('I', 16),
            ('F', 17),
            ('Y', 18),
            ('W', 19),
            ('L', 20),
        ];
        assert_eq!(upstream.len(), VOCAB_SIZE);
        for (c, id) in upstream {
            assert_eq!(id_to_char(id), Some(c), "id {id}");
        }
    }

    #[test]
    fn test_paired_layout() -> Result<()> {
        let enc = encode_pair("EV", "DI")?;
        assert_eq!(decode(&enc.ids)?, "<EV>|<DI>");
        assert_eq!(enc.residue_rows(Chain::Heavy), vec![1, 2]);
        assert_eq!(enc.residue_rows(Chain::Light), vec![6, 7]);
        assert_eq!(
            enc.roles[6],
            TokenRole::Residue {
                chain: Chain::Light,
                index: 0
            }
        );
        Ok(())
    }

    #[test]
    fn test_single_chain_layouts_are_asymmetric() -> Result<()> {
        assert_eq!(decode(&encode_pair("EV", "")?.ids)?, "<EV>|");
        assert_eq!(decode(&encode_pair("", "DI")?.ids)?, "|<DI>");
        // The first light residue sits at row 2 alone but row 6 after "EV".
        assert_eq!(
            encode_pair("", "DI")?.residue_rows(Chain::Light),
            vec![2, 3]
        );
        Ok(())
    }

    #[test]
    fn test_mask_and_unknown_are_residues_but_not_scorable() -> Result<()> {
        let enc = encode_pair("E*X", "")?;
        assert_eq!(
            enc.ids,
            vec![START_ID, 6, MASK_ID, UNKNOWN_ID, END_ID, SEP_ID]
        );
        assert_eq!(enc.residue_rows(Chain::Heavy), vec![1, 2, 3]);
        assert_eq!(enc.scorable_rows(), vec![1]);
        Ok(())
    }

    #[test]
    fn test_structural_characters_are_rejected_in_chain_text() {
        for bad in ["E|V", "E<V", "E>V", "E-V", "ev", "E V", "EBV", "EZV", "EUV"] {
            assert!(encode_pair(bad, "").is_err(), "{bad:?} should be rejected");
            assert!(encode_pair("", bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn test_empty_pair_is_rejected() {
        assert!(encode_pair("", "").is_err());
    }

    #[test]
    fn test_split_pair_requires_exactly_one_separator() -> Result<()> {
        assert_eq!(split_pair("EV|DI")?, ("EV", "DI"));
        assert_eq!(split_pair("EV|")?, ("EV", ""));
        assert_eq!(split_pair("|DI")?, ("", "DI"));
        assert!(split_pair("EVDI").is_err());
        assert!(split_pair("E|V|DI").is_err());
        Ok(())
    }
}
