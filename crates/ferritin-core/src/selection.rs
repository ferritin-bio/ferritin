//! Typed, topology-only row selections. No coordinates, operators, or source atom IDs
//! participate in this cache. Bind results to a model only after checking
//! [`SelectionIndex::is_current_for`].

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::data::OrderedSet;
use crate::model::MISSING_SEQ_ID;
use crate::{AtomicHierarchy, Model};

/// Conjunctive selection predicates; absent fields impose no restriction.
/// Ranges are inclusive. Dense indices are zero-based; negative indices match
/// nothing. Identifiers are exact, case-sensitive strings (elements use the
/// hierarchy's uppercase symbols). An empty insertion code means no insertion.
/// Missing label sequence IDs never match a sequence predicate, even a range.
#[derive(Default, Clone, Debug)]
pub struct RowSelection {
    pub label_entity_id: Option<String>,
    pub label_asym_id: Option<String>,
    pub auth_asym_id: Option<String>,
    pub label_atom_id: Option<String>,
    pub auth_atom_id: Option<String>,
    pub type_symbol: Option<String>,
    pub label_seq_id: Option<i32>,
    pub auth_seq_id: Option<i32>,
    pub beg_label_seq_id: Option<i32>,
    pub end_label_seq_id: Option<i32>,
    pub beg_auth_seq_id: Option<i32>,
    pub end_auth_seq_id: Option<i32>,
    pub residue_index: Option<i32>,
    pub atom_index: Option<i32>,
    pub pdbx_pdb_ins_code: Option<String>,
}

type Postings<K> = BTreeMap<K, Vec<usize>>;

/// A pinned hierarchy snapshot with independently lazy chain/sequence postings.
/// Coordinate-only frames reuse it; `Arc::make_mut` topology edits detach from it.
/// All query scratch is local, so concurrent reads are supported.
pub struct SelectionIndex {
    hierarchy: Arc<AtomicHierarchy>,
    entity: OnceLock<Postings<String>>,
    label_chain: OnceLock<Postings<String>>,
    auth_chain: OnceLock<Postings<String>>,
    label_seq: OnceLock<Postings<i32>>,
    auth_seq: OnceLock<Postings<i32>>,
}

// Keep small structures (including 1D3Z) off the cold-index allocation path.
const SCAN_THRESHOLD: usize = 2048;

impl SelectionIndex {
    /// Pin the exact hierarchy allocation, without constructing any postings.
    pub fn new(model: &Model) -> Self {
        Self {
            hierarchy: Arc::clone(&model.hierarchy),
            entity: OnceLock::new(),
            label_chain: OnceLock::new(),
            auth_chain: OnceLock::new(),
            label_seq: OnceLock::new(),
            auth_seq: OnceLock::new(),
        }
    }

    /// Whether a model still shares this exact topology snapshot.
    pub fn is_current_for(&self, model: &Model) -> bool {
        Arc::ptr_eq(&self.hierarchy, &model.hierarchy)
    }

    /// Select ascending, unique atom indices in this snapshot, not a live model.
    /// Uses chain/residue candidates and atom filtering, never a full atom mask.
    /// Models with fewer than 2048 atoms use a hierarchical scan instead of
    /// constructing postings. Atom/residue index predicates take a direct path.
    pub fn select(&self, row: &RowSelection) -> OrderedSet {
        let h = &self.hierarchy;
        let mut atoms = Vec::new();
        if let Some(a) = row.atom_index {
            if a >= 0 && (a as usize) < h.n_atoms() {
                let a = a as usize;
                let r = h.residue_of_atom(a);
                if self.matches_residue(row, r) && self.matches_atom(row, a) {
                    atoms.push(a as u32);
                }
            }
            return OrderedSet::from_sorted(atoms);
        }
        let mut visit = |r| {
            if self.matches_residue(row, r) {
                for a in h.atoms_in_residue(r) {
                    if self.matches_atom(row, a) {
                        atoms.push(a as u32);
                    }
                }
            }
        };
        if let Some(r) = row.residue_index {
            if r >= 0 && (r as usize) < h.n_residues() {
                visit(r as usize);
            }
        } else if h.n_atoms() < SCAN_THRESHOLD {
            (0..h.n_residues()).for_each(&mut visit);
        } else if let Some(mut candidates) = self.candidates(row) {
            candidates.sort_unstable();
            candidates.into_iter().for_each(&mut visit);
        } else {
            (0..h.n_residues()).for_each(&mut visit);
        }
        OrderedSet::from_sorted(atoms)
    }

    /// One result per row, preserving row order and overlap between rows.
    pub fn select_many(&self, rows: &[RowSelection]) -> Vec<OrderedSet> {
        rows.iter().map(|row| self.select(row)).collect()
    }

    /// Estimated bytes of initialized postings only, excluding the pinned
    /// hierarchy, inline `Self`, results and query scratch. Counts key/vector
    /// storage and vector capacities, with three pointer words per B-tree entry
    /// as an approximate node overhead. Not an allocator/RSS measurement.
    pub fn index_bytes(&self) -> usize {
        let strings = [&self.entity, &self.label_chain, &self.auth_chain]
            .into_iter()
            .filter_map(OnceLock::get)
            .map(|map| estimate(map) + map.keys().map(String::capacity).sum::<usize>())
            .sum::<usize>();
        strings
            + [&self.label_seq, &self.auth_seq]
                .into_iter()
                .filter_map(OnceLock::get)
                .map(estimate)
                .sum::<usize>()
    }

    fn candidates(&self, row: &RowSelection) -> Option<Vec<usize>> {
        let h = &self.hierarchy;
        enum Candidate<'a> {
            Chains(&'a [usize]),
            Sequence(&'a Postings<i32>, i32, i32),
        }
        let mut best: Option<(usize, Candidate<'_>)> = None;
        // Choose the smallest available posting expansion; remaining predicates
        // are checked directly, so duplicate identifier keys retain all matches.
        for (exact, begin, end, lock, column, missing) in [
            (
                row.label_seq_id,
                row.beg_label_seq_id,
                row.end_label_seq_id,
                &self.label_seq,
                &h.residues.label_seq_id,
                true,
            ),
            (
                row.auth_seq_id,
                row.beg_auth_seq_id,
                row.end_auth_seq_id,
                &self.auth_seq,
                &h.residues.auth_seq_id,
                false,
            ),
        ] {
            if exact.is_some() || begin.is_some() || end.is_some() {
                let lo = exact.unwrap_or(i32::MIN).max(begin.unwrap_or(i32::MIN));
                let hi = exact.unwrap_or(i32::MAX).min(end.unwrap_or(i32::MAX));
                if lo > hi {
                    return Some(Vec::new());
                }
                let map = lock.get_or_init(|| {
                    postings(
                        column
                            .iter()
                            .copied()
                            .enumerate()
                            .filter(|&(_, v)| !missing || v != MISSING_SEQ_ID),
                    )
                });
                let count: usize = map.range(lo..=hi).map(|(_, ids)| ids.len()).sum();
                if best.as_ref().is_none_or(|(n, _)| count < *n) {
                    best = Some((count, Candidate::Sequence(map, lo, hi)));
                }
            }
        }
        for (value, lock, column) in [
            (&row.label_entity_id, &self.entity, &h.chains.entity_id),
            (
                &row.label_asym_id,
                &self.label_chain,
                &h.chains.label_asym_id,
            ),
            (&row.auth_asym_id, &self.auth_chain, &h.chains.auth_asym_id),
        ] {
            if let Some(value) = value {
                let map = lock.get_or_init(|| postings(column.iter().cloned().enumerate()));
                let chains = map.get(value).map(Vec::as_slice).unwrap_or_default();
                let count: usize = chains.iter().map(|&c| h.residues_in_chain(c).len()).sum();
                if best.as_ref().is_none_or(|(n, _)| count < *n) {
                    best = Some((count, Candidate::Chains(chains)));
                }
            }
        }
        // Expand only the winning candidate, not all matching postings.
        best.map(|(_, candidate)| match candidate {
            Candidate::Chains(chains) => chains
                .iter()
                .flat_map(|&c| h.residues_in_chain(c))
                .collect(),
            Candidate::Sequence(map, lo, hi) => map
                .range(lo..=hi)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect(),
        })
    }

    fn matches_residue(&self, row: &RowSelection, r: usize) -> bool {
        let h = &self.hierarchy;
        let c = h.chain_of_residue(r);
        let label = h.residues.label_seq_id[r];
        let label_restricted = row.label_seq_id.is_some()
            || row.beg_label_seq_id.is_some()
            || row.end_label_seq_id.is_some();
        row.residue_index.is_none_or(|v| v >= 0 && v as usize == r)
            && text_matches(&row.label_entity_id, &h.chains.entity_id[c])
            && text_matches(&row.label_asym_id, &h.chains.label_asym_id[c])
            && text_matches(&row.auth_asym_id, &h.chains.auth_asym_id[c])
            && (!label_restricted || label != MISSING_SEQ_ID)
            && number_matches(
                label,
                row.label_seq_id,
                row.beg_label_seq_id,
                row.end_label_seq_id,
            )
            && number_matches(
                h.residues.auth_seq_id[r],
                row.auth_seq_id,
                row.beg_auth_seq_id,
                row.end_auth_seq_id,
            )
            && row
                .pdbx_pdb_ins_code
                .as_ref()
                .is_none_or(|code| match h.residues.ins_code[r] {
                    None => code.is_empty(),
                    Some(c) => {
                        let mut chars = code.chars();
                        chars.next() == Some(c) && chars.next().is_none()
                    }
                })
    }

    fn matches_atom(&self, row: &RowSelection, a: usize) -> bool {
        let atoms = &self.hierarchy.atoms;
        text_matches(&row.label_atom_id, &atoms.atom_name[a])
            && text_matches(&row.auth_atom_id, &atoms.auth_atom_name[a])
            && text_matches(&row.type_symbol, atoms.element[a].symbol())
    }
}

fn text_matches(filter: &Option<String>, value: &str) -> bool {
    filter.as_ref().is_none_or(|s| s == value)
}

fn number_matches(value: i32, exact: Option<i32>, begin: Option<i32>, end: Option<i32>) -> bool {
    exact.is_none_or(|v| value == v)
        && begin.is_none_or(|v| value >= v)
        && end.is_none_or(|v| value <= v)
}

fn postings<K: Ord>(values: impl Iterator<Item = (usize, K)>) -> Postings<K> {
    let mut map = BTreeMap::new();
    for (i, value) in values {
        map.entry(value).or_insert_with(Vec::new).push(i);
    }
    map
}

fn estimate<K>(map: &Postings<K>) -> usize {
    map.len()
        * (std::mem::size_of::<K>()
            + std::mem::size_of::<Vec<usize>>()
            + 3 * std::mem::size_of::<usize>())
        + map
            .values()
            .map(|v| v.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
}
