//! Public-contract tests with an independent, atom-by-atom scan oracle.
use std::sync::Arc;

use ferritin_core::data::Segmentation;
use ferritin_core::info::elements::Element;
use ferritin_core::model::{AtomsTable, ChainsTable, MISSING_SEQ_ID, ResidueGroup, ResiduesTable};
use ferritin_core::selection::{RowSelection, SelectionIndex};
use ferritin_core::{AtomicConformation, AtomicHierarchy, Bonds, Model};

fn fixture(chains: usize) -> Model {
    let residues = chains * 4;
    let atoms = residues * 2;
    Model::new(
        Arc::new(AtomicHierarchy {
            atoms: AtomsTable {
                atom_name: (0..atoms)
                    .map(|a| if a % 2 == 0 { "CA" } else { "N" }.into())
                    .collect(),
                auth_atom_name: (0..atoms)
                    .map(|a| if a % 2 == 0 { "C1" } else { "N1" }.into())
                    .collect(),
                element: (0..atoms)
                    .map(|a| if a % 2 == 0 { Element::C } else { Element::N })
                    .collect(),
                alt_loc: vec![None; atoms],
                formal_charge: vec![None; atoms],
            },
            residues: ResiduesTable {
                comp_id: vec!["ALA".into(); residues],
                auth_comp_id: vec!["GLY".into(); residues],
                label_seq_id: (0..residues)
                    .map(|r| [MISSING_SEQ_ID, 7, 7, 9][r % 4])
                    .collect(),
                auth_seq_id: (0..residues).map(|r| [-2, 10, 10, 30][r % 4]).collect(),
                ins_code: (0..residues)
                    .map(|r| [None, None, Some('A'), Some('β')][r % 4])
                    .collect(),
                group: vec![ResidueGroup::Polymer; residues],
            },
            chains: ChainsTable {
                // Intentionally duplicated keys, with asymmetric namespaces.
                label_asym_id: (0..chains).map(|c| format!("L{}", c / 2)).collect(),
                auth_asym_id: (0..chains).map(|c| format!("X{}", c % 3)).collect(),
                entity_id: (0..chains).map(|c| format!("{}", c % 2)).collect(),
            },
            atom_to_residue: Segmentation::from_offsets(
                (0..=residues as u32).map(|r| r * 2).collect(),
            ),
            residue_to_chain: Segmentation::from_offsets(
                (0..=chains as u32).map(|c| c * 4).collect(),
            ),
            bonds: Bonds::from_unsorted(vec![], vec![], vec![], atoms),
        }),
        AtomicConformation {
            x: vec![0.; atoms],
            y: vec![0.; atoms],
            z: vec![0.; atoms],
            occupancy: None,
            b_iso: None,
            confidence: None,
        },
    )
}

fn scan(model: &Model, row: &RowSelection) -> Vec<u32> {
    let h = &model.hierarchy;
    (0..model.n_atoms())
        .filter(|&a| {
            let r = h.residue_of_atom(a);
            let c = h.chain_of_residue(r);
            let strings = [
                (&row.label_entity_id, h.chains.entity_id[c].as_str()),
                (&row.label_asym_id, h.chains.label_asym_id[c].as_str()),
                (&row.auth_asym_id, h.chains.auth_asym_id[c].as_str()),
                (&row.label_atom_id, h.atoms.atom_name[a].as_str()),
                (&row.auth_atom_id, h.atoms.auth_atom_name[a].as_str()),
                (&row.type_symbol, h.atoms.element[a].symbol()),
            ];
            for (wanted, actual) in strings {
                if let Some(wanted) = wanted
                    && wanted != actual
                {
                    return false;
                }
            }
            for (wanted, actual) in [(&row.atom_index, a), (&row.residue_index, r)] {
                if let Some(wanted) = wanted
                    && usize::try_from(*wanted).ok() != Some(actual)
                {
                    return false;
                }
            }
            for (exact, begin, end, actual, missing) in [
                (
                    row.label_seq_id,
                    row.beg_label_seq_id,
                    row.end_label_seq_id,
                    h.residues.label_seq_id[r],
                    true,
                ),
                (
                    row.auth_seq_id,
                    row.beg_auth_seq_id,
                    row.end_auth_seq_id,
                    h.residues.auth_seq_id[r],
                    false,
                ),
            ] {
                if missing
                    && actual == MISSING_SEQ_ID
                    && (exact.is_some() || begin.is_some() || end.is_some())
                {
                    return false;
                }
                if let Some(v) = exact
                    && actual != v
                {
                    return false;
                }
                if let Some(v) = begin
                    && actual < v
                {
                    return false;
                }
                if let Some(v) = end
                    && actual > v
                {
                    return false;
                }
            }
            if let Some(code) = &row.pdbx_pdb_ins_code
                && *code
                    != h.residues.ins_code[r]
                        .map(|c| c.to_string())
                        .unwrap_or_default()
            {
                return false;
            }
            true
        })
        .map(|a| a as u32)
        .collect()
}

fn rows() -> Vec<RowSelection> {
    let mut rows = vec![RowSelection::default()];
    for value in [MISSING_SEQ_ID, -3, -2, 0, 7, 9, 10, 30, i32::MAX] {
        rows.extend([
            RowSelection {
                label_seq_id: Some(value),
                ..Default::default()
            },
            RowSelection {
                auth_seq_id: Some(value),
                ..Default::default()
            },
            RowSelection {
                beg_label_seq_id: Some(value),
                ..Default::default()
            },
            RowSelection {
                end_label_seq_id: Some(value),
                ..Default::default()
            },
            RowSelection {
                beg_auth_seq_id: Some(value),
                ..Default::default()
            },
            RowSelection {
                end_auth_seq_id: Some(value),
                ..Default::default()
            },
            RowSelection {
                atom_index: Some(value),
                ..Default::default()
            },
            RowSelection {
                residue_index: Some(value),
                ..Default::default()
            },
        ]);
    }
    for name in [
        "L0", "X0", "0", "CA", "C1", "N", "N1", "", "A", "β", "AA", "unknown",
    ] {
        rows.extend([
            RowSelection {
                label_entity_id: Some(name.into()),
                ..Default::default()
            },
            RowSelection {
                label_asym_id: Some(name.into()),
                ..Default::default()
            },
            RowSelection {
                auth_asym_id: Some(name.into()),
                ..Default::default()
            },
            RowSelection {
                label_atom_id: Some(name.into()),
                ..Default::default()
            },
            RowSelection {
                auth_atom_id: Some(name.into()),
                ..Default::default()
            },
            RowSelection {
                type_symbol: Some(name.into()),
                ..Default::default()
            },
            RowSelection {
                pdbx_pdb_ins_code: Some(name.into()),
                ..Default::default()
            },
        ]);
    }
    for i in 0..120 {
        rows.push(RowSelection {
            label_entity_id: Some((i % 2).to_string()),
            label_asym_id: Some(format!("L{}", i % 4)),
            auth_asym_id: Some(format!("X{}", i % 3)),
            label_atom_id: Some(if i % 2 == 0 { "CA" } else { "N" }.into()),
            auth_atom_id: Some(if i % 2 == 0 { "C1" } else { "N1" }.into()),
            type_symbol: Some(if i % 2 == 0 { "C" } else { "N" }.into()),
            beg_label_seq_id: Some([7, 9, 10][i % 3]),
            end_label_seq_id: Some([9, 7, 30][i % 3]),
            auth_seq_id: Some([10, 30, -2][i % 3]),
            pdbx_pdb_ins_code: Some(["", "A", "β"][i % 3].into()),
            ..Default::default()
        });
    }
    rows
}

#[test]
fn test_selection_scan_and_index_match_independent_oracle() {
    for count in [0, 2, 255, 256, 257] {
        let model = fixture(count);
        let index = SelectionIndex::new(&model);
        assert_eq!(index.index_bytes(), 0);
        let rows = rows();
        let selected = index.select_many(&rows);
        for (row, result) in rows.iter().zip(selected) {
            assert_eq!(
                result.iter().collect::<Vec<_>>(),
                scan(&model, row),
                "{row:?}"
            );
        }
        if count < 256 {
            assert_eq!(index.index_bytes(), 0);
        } else {
            assert!(index.index_bytes() > 0);
        }
    }
}

#[test]
fn test_selection_duplicates_overlap_order_and_missing() {
    let model = fixture(256);
    let index = SelectionIndex::new(&model);
    let duplicate = RowSelection {
        label_asym_id: Some("L0".into()),
        label_seq_id: Some(7),
        ..Default::default()
    };
    let single = RowSelection {
        atom_index: Some(2),
        ..Default::default()
    };
    let selected = index.select_many(&[duplicate.clone(), single, duplicate]);
    assert_eq!(
        selected[0].iter().collect::<Vec<_>>(),
        vec![2, 3, 4, 5, 10, 11, 12, 13]
    );
    assert_eq!(selected[1].iter().collect::<Vec<_>>(), vec![2]);
    assert_eq!(selected[0], selected[2]);
    assert_eq!(
        index
            .select(&RowSelection {
                beg_label_seq_id: Some(i32::MIN),
                end_label_seq_id: Some(i32::MAX),
                ..Default::default()
            })
            .len(),
        model.n_atoms() * 3 / 4
    );
    assert!(
        index
            .select(&RowSelection {
                label_seq_id: Some(MISSING_SEQ_ID),
                ..Default::default()
            })
            .is_empty()
    );
    assert!(
        index
            .select(&RowSelection {
                label_seq_id: Some(7),
                beg_label_seq_id: Some(8),
                ..Default::default()
            })
            .is_empty()
    );
}

#[test]
fn test_selection_coordinate_reuse_and_topology_snapshot_isolation() {
    let mut model = fixture(256);
    let index = SelectionIndex::new(&model);
    let row = RowSelection {
        label_seq_id: Some(7),
        ..Default::default()
    };
    let before = index.select(&row);
    let bytes = index.index_bytes();
    model.conformation.x.fill(42.);
    assert!(index.is_current_for(&model));
    assert_eq!(before, index.select(&row));
    assert_eq!(bytes, index.index_bytes());
    let frame = model.clone();
    Arc::make_mut(&mut model.hierarchy)
        .residues
        .label_seq_id
        .fill(99);
    assert!(!index.is_current_for(&model));
    assert!(index.is_current_for(&frame));
    assert_eq!(before, index.select(&row));
    assert!(SelectionIndex::new(&model).select(&row).is_empty());
    // Also detach before first lazy initialization.
    let cold = SelectionIndex::new(&model);
    Arc::make_mut(&mut model.hierarchy)
        .residues
        .label_seq_id
        .fill(7);
    assert!(!cold.is_current_for(&model));
    assert!(cold.select(&row).is_empty());
}

#[test]
fn test_selection_concurrent_cold_and_warm_reads() {
    let model = fixture(256);
    let index = SelectionIndex::new(&model);
    let rows = rows();
    let expected: Vec<_> = rows.iter().map(|row| scan(&model, row)).collect();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..2 {
                    for (result, expected) in index.select_many(&rows).iter().zip(&expected) {
                        assert_eq!(result.iter().collect::<Vec<_>>(), *expected);
                    }
                }
            });
        }
    });
}

/// Replicate real topology/coordinates, renaming chains to address each copy.
fn replicate_model(source: &Model, copies: usize) -> Model {
    let original = &source.hierarchy;
    let mut h = (**original).clone();
    macro_rules! extend_columns {
        ($table:ident, $($field:ident),+) => {
            $(for _ in 1..copies {
                h.$table.$field.extend_from_slice(&original.$table.$field);
            })+
        };
    }
    extend_columns!(
        atoms,
        atom_name,
        auth_atom_name,
        element,
        alt_loc,
        formal_charge
    );
    extend_columns!(
        residues,
        comp_id,
        auth_comp_id,
        label_seq_id,
        auth_seq_id,
        ins_code,
        group
    );
    extend_columns!(chains, label_asym_id, auth_asym_id, entity_id);
    for c in 0..h.n_chains() {
        let copy = c / source.n_chains();
        h.chains.label_asym_id[c].push_str(&format!(":{copy}"));
        h.chains.auth_asym_id[c].push_str(&format!(":{copy}"));
    }
    let mut atom_offsets = vec![0];
    let mut residue_offsets = vec![0];
    for copy in 0..copies {
        atom_offsets.extend(
            original.atom_to_residue.offsets()[1..]
                .iter()
                .map(|v| v + (copy * source.n_atoms()) as u32),
        );
        residue_offsets.extend(
            original.residue_to_chain.offsets()[1..]
                .iter()
                .map(|v| v + (copy * source.n_residues()) as u32),
        );
    }
    h.atom_to_residue = Segmentation::from_offsets(atom_offsets);
    h.residue_to_chain = Segmentation::from_offsets(residue_offsets);
    h.bonds = Bonds::from_unsorted(
        (0..copies)
            .flat_map(|copy| {
                original
                    .bonds
                    .atom_a
                    .iter()
                    .map(move |a| a + (copy * source.n_atoms()) as u32)
            })
            .collect(),
        (0..copies)
            .flat_map(|copy| {
                original
                    .bonds
                    .atom_b
                    .iter()
                    .map(move |a| a + (copy * source.n_atoms()) as u32)
            })
            .collect(),
        (0..copies)
            .flat_map(|_| original.bonds.order.iter().copied())
            .collect(),
        source.n_atoms() * copies,
    );
    let mut conformation = source.conformation.clone();
    for _ in 1..copies {
        conformation.x.extend_from_slice(&source.conformation.x);
        conformation.y.extend_from_slice(&source.conformation.y);
        conformation.z.extend_from_slice(&source.conformation.z);
    }
    // These optional columns do not participate in selection timings.
    conformation.occupancy = None;
    conformation.b_iso = None;
    conformation.confidence = None;
    Model::new(Arc::new(h), conformation)
}

/// Independent hierarchical baseline for the benchmark's chain/sequence/name
/// rows. Checks chain once per chain and sequence once per residue, rather than
/// using the deliberately simple atom-by-atom contract oracle above.
fn benchmark_scan(model: &Model, row: &RowSelection) -> Vec<u32> {
    let h = &model.hierarchy;
    let mut selected = Vec::new();
    for c in 0..h.n_chains() {
        if row.label_asym_id.as_ref() != Some(&h.chains.label_asym_id[c]) {
            continue;
        }
        for r in h.residues_in_chain(c) {
            if h.residues.label_seq_id[r] == MISSING_SEQ_ID
                || row.label_seq_id != Some(h.residues.label_seq_id[r])
            {
                continue;
            }
            for a in h.atoms_in_residue(r) {
                if row.label_atom_id.as_ref() == Some(&h.atoms.atom_name[a]) {
                    selected.push(a as u32);
                }
            }
        }
    }
    selected
}

/// Reproducible realistic algorithm benchmark, not an MVS end-to-end claim.
/// Run: cargo test -p ferritin-core --release --test test_selection benchmark_selection -- --ignored --nocapture
#[test]
#[ignore = "release-mode timing diagnostic"]
fn benchmark_selection_cold_warm_memory() {
    use ferritin_test_data::TestFile;
    use std::hint::black_box;
    use std::time::Instant;
    for (file, copies) in [
        (TestFile::multimodel_nmr_1d3z(), 1),
        (TestFile::mvs_1c0a(), 20),
    ] {
        let (path, _handle) = file.create_temp().unwrap();
        let source = ferritin_core::load_model(&path).unwrap();
        let model = replicate_model(&source, copies);
        let h = &model.hierarchy;
        let polymer: Vec<_> = (0..model.n_residues())
            .filter(|&r| h.residues.label_seq_id[r] != MISSING_SEQ_ID)
            .collect();
        for count in [1, 1000] {
            let rows: Vec<_> = (0..count)
                .map(|i| {
                    let r = polymer[(i * 37) % polymer.len()];
                    let a = h.atoms_in_residue(r).start;
                    RowSelection {
                        label_asym_id: Some(h.chains.label_asym_id[h.chain_of_residue(r)].clone()),
                        label_seq_id: Some(h.residues.label_seq_id[r]),
                        label_atom_id: Some(h.atoms.atom_name[a].clone()),
                        ..Default::default()
                    }
                })
                .collect();
            let start = Instant::now();
            let expected: Vec<_> = rows.iter().map(|row| benchmark_scan(&model, row)).collect();
            let scan_time = start.elapsed();
            let start = Instant::now();
            let index = SelectionIndex::new(&model);
            let creation_time = start.elapsed();
            assert_eq!(index.index_bytes(), 0);
            let start = Instant::now();
            let cold = index.select_many(&rows);
            let cold_time = start.elapsed();
            for (result, expected) in cold.iter().zip(&expected) {
                assert_eq!(result.iter().collect::<Vec<_>>(), *expected);
            }
            let mut warm = Vec::new();
            for _ in 0..7 {
                let start = Instant::now();
                let results = black_box(index.select_many(black_box(&rows)));
                warm.push(start.elapsed());
                for (result, expected) in results.iter().zip(&expected) {
                    assert_eq!(result.iter().collect::<Vec<_>>(), *expected);
                }
            }
            warm.sort_unstable();
            eprintln!(
                "atoms={} rows={count} hierarchical_scan={scan_time:?} creation={creation_time:?} cold_query={cold_time:?} warm_median={:?} index_bytes_estimate={}",
                model.n_atoms(),
                warm[3],
                index.index_bytes()
            );
        }
    }
}
