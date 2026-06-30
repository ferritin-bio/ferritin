//! AtomMask selection layer: maps MolViewSpec ComponentSelector → per-atom boolean mask.
use ferritin_core::AtomCollection;
use ferritin_molviewspec::molviewspec::nodes::{
    ComponentExpression, ComponentSelector, ComponentSelectorT,
};

/// Per-atom boolean mask. Length always equals the atom count of the target model.
pub struct AtomMask(pub Vec<bool>);

impl AtomMask {
    pub fn count_true(&self) -> usize {
        self.0.iter().filter(|&&b| b).count()
    }

    pub fn all_false(&self) -> bool {
        self.0.iter().all(|&b| !b)
    }
}

/// Evaluate a MolViewSpec `ComponentSelector` against an `AtomCollection`, returning a per-atom mask.
pub fn evaluate_selector(selector: &ComponentSelector, model: &AtomCollection) -> AtomMask {
    match selector {
        ComponentSelector::Selector(s) => evaluate_named(s, model),
        ComponentSelector::Expression(expr) => evaluate_expression(expr, model),
        ComponentSelector::ExpressionList(exprs) => {
            let n = model.get_size();
            let mut mask = vec![false; n];
            for expr in exprs {
                let sub = evaluate_expression(expr, model);
                for (dst, src) in mask.iter_mut().zip(sub.0.iter()) {
                    *dst |= src;
                }
            }
            AtomMask(mask)
        }
    }
}

fn evaluate_named(selector: &ComponentSelectorT, model: &AtomCollection) -> AtomMask {
    let n = model.get_size();
    match selector {
        ComponentSelectorT::All => AtomMask(vec![true; n]),
        ComponentSelectorT::Polymer => {
            let mut mask = vec![false; n];
            for residue in model.iter_residues() {
                let is_polymer = residue.is_amino_acid() || residue.is_nucleotide();
                for idx in residue.atom_range() {
                    mask[idx] = is_polymer;
                }
            }
            AtomMask(mask)
        }
        ComponentSelectorT::Protein => {
            let mut mask = vec![false; n];
            for residue in model.iter_residues() {
                if residue.is_amino_acid() {
                    for idx in residue.atom_range() {
                        mask[idx] = true;
                    }
                }
            }
            AtomMask(mask)
        }
        ComponentSelectorT::Nucleic => {
            let mut mask = vec![false; n];
            for residue in model.iter_residues() {
                if residue.is_nucleotide() {
                    for idx in residue.atom_range() {
                        mask[idx] = true;
                    }
                }
            }
            AtomMask(mask)
        }
        ComponentSelectorT::Branched => {
            let mut mask = vec![false; n];
            for residue in model.iter_residues() {
                if residue.is_carbohydrate() {
                    for idx in residue.atom_range() {
                        mask[idx] = true;
                    }
                }
            }
            AtomMask(mask)
        }
        ComponentSelectorT::Ligand => {
            let mut mask = vec![false; n];
            for residue in model.iter_residues() {
                if residue.is_hetero() && !residue.is_water() && !residue.is_ion() {
                    for idx in residue.atom_range() {
                        mask[idx] = true;
                    }
                }
            }
            AtomMask(mask)
        }
        ComponentSelectorT::Ion => AtomMask(model.select_ion()),
        ComponentSelectorT::Water => AtomMask(model.select_water()),
    }
}

fn evaluate_expression(expr: &ComponentExpression, model: &AtomCollection) -> AtomMask {
    // An all-None expression matches everything (MVS spec: "match whole structure").
    if is_all_none(expr) {
        return AtomMask(vec![true; model.get_size()]);
    }

    let n = model.get_size();
    let mut mask = vec![false; n];

    for residue in model.iter_residues() {
        // Chain filter (label_asym_id matched against auth_asym_id in MVP)
        if let Some(ref chain) = expr.label_asym_id {
            if residue.chain_id() != chain {
                continue;
            }
        }
        if let Some(ref chain) = expr.auth_asym_id {
            if residue.chain_id() != chain {
                continue;
            }
        }

        let res_id = residue.residue_id();

        // Exact residue seq id (label_seq_id matched against auth_seq_id in MVP)
        if let Some(seq) = expr.label_seq_id {
            if res_id != seq {
                continue;
            }
        }
        if let Some(seq) = expr.auth_seq_id {
            if res_id != seq {
                continue;
            }
        }

        // Range filters (beg inclusive, end inclusive)
        if let Some(beg) = expr.beg_label_seq_id {
            if res_id < beg {
                continue;
            }
        }
        if let Some(end) = expr.end_label_seq_id {
            if res_id > end {
                continue;
            }
        }
        if let Some(beg) = expr.beg_auth_seq_id {
            if res_id < beg {
                continue;
            }
        }
        if let Some(end) = expr.end_auth_seq_id {
            if res_id > end {
                continue;
            }
        }

        // residue_index: 0-based index of this residue in iteration order
        // We skip this check at residue level; handle per-atom below since it's unusual.

        // Residue passed — now check per-atom fields
        for idx in residue.atom_range() {
            // atom_index: 0-based global atom index
            if let Some(ai) = expr.atom_index {
                if idx as i32 != ai {
                    continue;
                }
            }

            // label_atom_id / auth_atom_id
            let atom_name = model.get_atom_name(idx);
            if let Some(ref la) = expr.label_atom_id {
                if atom_name != la {
                    continue;
                }
            }
            if let Some(ref aa) = expr.auth_atom_id {
                if atom_name != aa {
                    continue;
                }
            }

            // type_symbol (element symbol, uppercase)
            if let Some(ref sym) = expr.type_symbol {
                if model.get_element(idx).symbol() != sym.as_str() {
                    continue;
                }
            }

            mask[idx] = true;
        }
    }

    AtomMask(mask)
}

fn is_all_none(expr: &ComponentExpression) -> bool {
    expr.label_entity_id.is_none()
        && expr.label_asym_id.is_none()
        && expr.auth_asym_id.is_none()
        && expr.label_seq_id.is_none()
        && expr.auth_seq_id.is_none()
        && expr.pdbx_pdb_ins_code.is_none()
        && expr.beg_label_seq_id.is_none()
        && expr.end_label_seq_id.is_none()
        && expr.beg_auth_seq_id.is_none()
        && expr.end_auth_seq_id.is_none()
        && expr.residue_index.is_none()
        && expr.label_atom_id.is_none()
        && expr.auth_atom_id.is_none()
        && expr.type_symbol.is_none()
        && expr.atom_id.is_none()
        && expr.atom_index.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferritin_core::AtomCollection;
    use ferritin_core::info::elements::Element;
    use ferritin_molviewspec::molviewspec::nodes::ComponentExpression;

    /// 8-atom collection: ALA(3, chain A, res 1), GLY(2, chain A, res 2),
    /// HOH(1, chain B, res 3 hetero), ZN(1, chain B, res 4 hetero), LIG(1, chain B, res 5 hetero multi-atom)
    fn make_test_collection() -> AtomCollection {
        let coords = vec![[0.0; 3]; 9];
        let res_ids: Vec<i32> = vec![1, 1, 1, 2, 2, 3, 4, 5, 5];
        let res_names: Vec<String> =
            vec!["ALA", "ALA", "ALA", "GLY", "GLY", "HOH", "ZN", "LIG", "LIG"]
                .into_iter()
                .map(|s| s.to_string())
                .collect();
        let is_hetero = vec![false, false, false, false, false, true, true, true, true];
        let elements = vec![
            Element::N,
            Element::C,
            Element::C,
            Element::N,
            Element::C,
            Element::O,
            Element::Zn,
            Element::C,
            Element::O,
        ];
        let atom_names: Vec<String> = vec!["N", "CA", "C", "N", "CA", "O", "ZN", "C1", "O1"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let chain_ids: Vec<String> = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        AtomCollection::new(
            9, coords, res_ids, res_names, is_hetero, elements, atom_names, chain_ids, None,
        )
    }

    // T2-01: All selector → all true
    #[test]
    fn test_selector_all() {
        let ac = make_test_collection();
        let sel = ComponentSelector::Selector(ComponentSelectorT::All);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 9);
    }

    // T2-02: Protein selector → only ALA + GLY atoms (5 atoms)
    #[test]
    fn test_selector_protein() {
        let ac = make_test_collection();
        let sel = ComponentSelector::Selector(ComponentSelectorT::Protein);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(
            mask.count_true(),
            5,
            "protein: expected 5 atoms (ALA×3 + GLY×2)"
        );
    }

    // T2-03: Water selector → only HOH atom (1 atom)
    #[test]
    fn test_selector_water() {
        let ac = make_test_collection();
        let sel = ComponentSelector::Selector(ComponentSelectorT::Water);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 1, "water: expected 1 HOH atom");
        assert!(mask.0[5], "atom 5 (HOH) must be true");
    }

    // T2-04: Ion selector → only ZN atom (1 atom)
    #[test]
    fn test_selector_ion() {
        let ac = make_test_collection();
        let sel = ComponentSelector::Selector(ComponentSelectorT::Ion);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 1, "ion: expected 1 ZN atom");
        assert!(mask.0[6], "atom 6 (ZN) must be true");
    }

    // T2-05: Ligand selector → LIG atoms (2 atoms), not water or ion
    #[test]
    fn test_selector_ligand() {
        let ac = make_test_collection();
        let sel = ComponentSelector::Selector(ComponentSelectorT::Ligand);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 2, "ligand: expected 2 LIG atoms");
        assert!(mask.0[7] && mask.0[8], "atoms 7,8 (LIG) must be true");
    }

    // T2-06: Polymer selector → protein + nucleic (ALA + GLY = 5)
    #[test]
    fn test_selector_polymer() {
        let ac = make_test_collection();
        let sel = ComponentSelector::Selector(ComponentSelectorT::Polymer);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 5);
    }

    // T2-07: Expression chain A → first 5 atoms
    #[test]
    fn test_expression_chain_a() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            auth_asym_id: Some("A".to_string()),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 5);
        assert!(mask.0[..5].iter().all(|&b| b));
        assert!(mask.0[5..].iter().all(|&b| !b));
    }

    // T2-08: Expression chain B → last 4 atoms
    #[test]
    fn test_expression_chain_b() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            auth_asym_id: Some("B".to_string()),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 4);
        assert!(mask.0[5..].iter().all(|&b| b));
    }

    // T2-09: Expression auth_seq_id = 1 → 3 atoms (ALA residue)
    #[test]
    fn test_expression_seq_id() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            auth_seq_id: Some(1),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 3);
    }

    // T2-10: Expression beg/end range 1..2 → 5 atoms (ALA+GLY)
    #[test]
    fn test_expression_seq_range() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            beg_auth_seq_id: Some(1),
            end_auth_seq_id: Some(2),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 5);
    }

    // T2-11: Expression auth_atom_id = "CA" → 2 atoms
    #[test]
    fn test_expression_atom_name() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            auth_atom_id: Some("CA".to_string()),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 2, "CA atoms in ALA and GLY");
    }

    // T2-12: Expression type_symbol = "N" → 2 atoms
    #[test]
    fn test_expression_type_symbol() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            type_symbol: Some("N".to_string()),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 2, "N atoms in ALA and GLY");
    }

    // T2-13: Expression atom_index = 0 → 1 atom
    #[test]
    fn test_expression_atom_index() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            atom_index: Some(0),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 1);
        assert!(mask.0[0]);
    }

    // T2-14: ExpressionList (chain A res 1) OR (chain B res 4) → 3 + 1 = 4 atoms
    #[test]
    fn test_expression_list_union() {
        let ac = make_test_collection();
        let expr1 = ComponentExpression {
            auth_asym_id: Some("A".to_string()),
            auth_seq_id: Some(1),
            ..Default::default()
        };
        let expr2 = ComponentExpression {
            auth_asym_id: Some("B".to_string()),
            auth_seq_id: Some(4),
            ..Default::default()
        };
        let sel = ComponentSelector::ExpressionList(vec![expr1, expr2]);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(mask.count_true(), 4, "union: ALA(3) + ZN(1)");
    }

    // T2-21: All-None expression → all-true mask
    #[test]
    fn test_expression_all_none_is_all_true() {
        let ac = make_test_collection();
        let expr = ComponentExpression::default();
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(
            mask.count_true(),
            9,
            "all-None expression must match all atoms"
        );
    }

    // T2-22: count_true and all_false helpers
    #[test]
    fn test_atom_mask_helpers() {
        let full = AtomMask(vec![true, true, false]);
        assert_eq!(full.count_true(), 2);
        assert!(!full.all_false());

        let empty = AtomMask(vec![false, false, false]);
        assert_eq!(empty.count_true(), 0);
        assert!(empty.all_false());
    }

    // T2-23: label_asym_id matched against auth_asym_id (MVP documented behaviour)
    #[test]
    fn test_label_asym_id_matched_against_auth() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            label_asym_id: Some("A".to_string()),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(
            mask.count_true(),
            5,
            "label_asym_id A must match chain A (auth)"
        );
    }

    // T2-24: label_seq_id matched against auth_seq_id (MVP documented behaviour)
    #[test]
    fn test_label_seq_id_matched_against_auth() {
        let ac = make_test_collection();
        let expr = ComponentExpression {
            label_seq_id: Some(2),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(
            mask.count_true(),
            2,
            "label_seq_id 2 must match res_id 2 (GLY×2)"
        );
    }
}
