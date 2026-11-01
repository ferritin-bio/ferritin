//! MolViewSpec selectors backed by hierarchy-scoped core row selections.
use ferritin_core::selection::{RowSelection, SelectionIndex};
use ferritin_core::{AtomCollection, Model};
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

/// Compatibility entry point for flat collections, which have no distinct auth/label IDs.
///
/// Prefer [`evaluate_model_selector`] with the original model to preserve identifiers
/// and reuse indices. Panics on unsupported source `atom_id` selectors.
pub fn evaluate_selector(selector: &ComponentSelector, model: &AtomCollection) -> AtomMask {
    let model = model.to_model();
    let index = SelectionIndex::new(&model);
    evaluate_model_selector(selector, &model, &index).expect("unsupported MVS selector")
}

/// Evaluate against the original hierarchy, using one final mask for an entire row list.
///
/// Rows are ORed, fields within each row are ANDed. Source atom IDs are not
/// stored in the hierarchy and are rejected rather than treated as dense indices.
pub fn evaluate_model_selector(
    selector: &ComponentSelector,
    model: &Model,
    index: &SelectionIndex,
) -> anyhow::Result<AtomMask> {
    anyhow::ensure!(
        index.is_current_for(model),
        "selection index belongs to another hierarchy"
    );
    let rows = match selector {
        ComponentSelector::Selector(s) => {
            return Ok(evaluate_named(s, model));
        }
        ComponentSelector::Expression(expr) => vec![row_selection(expr)?],
        ComponentSelector::ExpressionList(exprs) => exprs
            .iter()
            .map(row_selection)
            .collect::<anyhow::Result<Vec<_>>>()?,
    };
    let mut mask = vec![false; model.n_atoms()];
    for row in index.select_many(&rows) {
        for atom in row.iter() {
            mask[atom as usize] = true;
        }
    }
    Ok(AtomMask(mask))
}

fn row_selection(expr: &ComponentExpression) -> anyhow::Result<RowSelection> {
    anyhow::ensure!(
        expr.atom_id.is_none(),
        "source atom_id selection is unsupported: source IDs are not stored; use atom_index for dense indices"
    );
    Ok(RowSelection {
        label_entity_id: expr.label_entity_id.clone(),
        label_asym_id: expr.label_asym_id.clone(),
        auth_asym_id: expr.auth_asym_id.clone(),
        label_seq_id: expr.label_seq_id,
        auth_seq_id: expr.auth_seq_id,
        pdbx_pdb_ins_code: expr.pdbx_pdb_ins_code.clone(),
        beg_label_seq_id: expr.beg_label_seq_id,
        end_label_seq_id: expr.end_label_seq_id,
        beg_auth_seq_id: expr.beg_auth_seq_id,
        end_auth_seq_id: expr.end_auth_seq_id,
        residue_index: expr.residue_index,
        label_atom_id: expr.label_atom_id.clone(),
        auth_atom_id: expr.auth_atom_id.clone(),
        type_symbol: expr.type_symbol.clone(),
        atom_index: expr.atom_index,
    })
}

fn evaluate_named(selector: &ComponentSelectorT, model: &Model) -> AtomMask {
    if matches!(selector, ComponentSelectorT::All) {
        return AtomMask(vec![true; model.n_atoms()]);
    }
    let mut mask = vec![false; model.n_atoms()];
    for (r, residue) in model.residues().enumerate() {
        let water = matches!(residue.residue_name(), "HOH" | "WAT" | "H2O");
        let hetero =
            model.hierarchy.residues.group[r] == ferritin_core::model::ResidueGroup::NonPolymer;
        let ion = hetero && residue.atom_count() == 1 && !water;
        let selected = match selector {
            ComponentSelectorT::All => true,
            ComponentSelectorT::Polymer => residue.is_amino_acid() || residue.is_nucleotide(),
            ComponentSelectorT::Protein => residue.is_amino_acid(),
            ComponentSelectorT::Nucleic => residue.is_nucleotide(),
            ComponentSelectorT::Branched => residue.is_carbohydrate(),
            ComponentSelectorT::Ligand => hetero && !water && !ion,
            ComponentSelectorT::Ion => ion,
            ComponentSelectorT::Water => water,
        };
        if selected {
            mask[model.hierarchy.atoms_in_residue(r)].fill(true);
        }
    }
    AtomMask(mask)
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

    // T2-25: residue_index (0-based) selects exactly the Nth residue in iteration order
    #[test]
    fn test_expression_residue_index() {
        let ac = make_test_collection();
        // residue_index=0 → ALA (3 atoms)
        let expr = ComponentExpression {
            residue_index: Some(0),
            ..Default::default()
        };
        let sel = ComponentSelector::Expression(expr);
        let mask = evaluate_selector(&sel, &ac);
        assert_eq!(
            mask.count_true(),
            3,
            "residue_index 0 must select ALA (3 atoms)"
        );
        assert!(
            mask.0[0] && mask.0[1] && mask.0[2],
            "atoms 0-2 should be selected"
        );

        // residue_index=1 → GLY (2 atoms)
        let expr2 = ComponentExpression {
            residue_index: Some(1),
            ..Default::default()
        };
        let mask2 = evaluate_selector(&ComponentSelector::Expression(expr2), &ac);
        assert_eq!(
            mask2.count_true(),
            2,
            "residue_index 1 must select GLY (2 atoms)"
        );

        // residue_index=3 → ZN (1 atom); residues: 0=ALA, 1=GLY, 2=HOH, 3=ZN, 4=LIG
        let expr3 = ComponentExpression {
            residue_index: Some(3),
            ..Default::default()
        };
        let mask3 = evaluate_selector(&ComponentSelector::Expression(expr3), &ac);
        assert_eq!(
            mask3.count_true(),
            1,
            "residue_index 3 must select ZN (1 atom)"
        );
    }

    // T2-26: residue_index combined with atom_name → single atom
    #[test]
    fn test_expression_residue_index_with_atom_name() {
        let ac = make_test_collection();
        // Select only the CA of the second residue (GLY)
        let expr = ComponentExpression {
            residue_index: Some(1),
            auth_atom_id: Some("CA".to_string()),
            ..Default::default()
        };
        let mask = evaluate_selector(&ComponentSelector::Expression(expr), &ac);
        assert_eq!(
            mask.count_true(),
            1,
            "residue 1 CA should be exactly 1 atom"
        );
        assert!(mask.0[4], "atom index 4 is GLY CA");
    }

    #[test]
    fn test_model_rows_preserve_identifiers_and_color_override_order() {
        use crate::colors::{VertexMapKind, apply_model_colors_with_theme};
        use bevy::asset::RenderAssetUsages;
        use bevy::prelude::Mesh;
        use bevy::render::mesh::{PrimitiveTopology, VertexAttributeValues};
        use ferritin_molviewspec::molviewspec::nodes::{ColorT, ColorThemeT};
        use std::sync::Arc;

        let mut model = make_test_collection().to_model();
        let hierarchy = Arc::make_mut(&mut model.hierarchy);
        hierarchy.chains.label_asym_id[0] = "LABEL".into();
        hierarchy.chains.entity_id[0] = "ENTITY".into();
        hierarchy.residues.label_seq_id[0] = 42;
        hierarchy.residues.ins_code[0] = Some('B');
        hierarchy.atoms.auth_atom_name[1] = "AUTH_CA".into();
        let index = SelectionIndex::new(&model);
        let row = ComponentExpression {
            label_asym_id: Some("LABEL".into()),
            auth_asym_id: Some("A".into()),
            label_entity_id: Some("ENTITY".into()),
            label_seq_id: Some(42),
            auth_seq_id: Some(1),
            pdbx_pdb_ins_code: Some("B".into()),
            label_atom_id: Some("CA".into()),
            auth_atom_id: Some("AUTH_CA".into()),
            ..Default::default()
        };
        let selector = ComponentSelector::ExpressionList(vec![row.clone(), row.clone()]);
        let mask = evaluate_model_selector(&selector, &model, &index).unwrap();
        assert_eq!(
            mask.0,
            vec![false, true, false, false, false, false, false, false, false]
        );
        let wrong = ComponentSelector::Expression(ComponentExpression {
            label_asym_id: Some("A".into()),
            ..Default::default()
        });
        assert!(
            evaluate_model_selector(&wrong, &model, &index)
                .unwrap()
                .all_false()
        );

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        apply_model_colors_with_theme(
            &mut mesh,
            &ColorThemeT::Uniform,
            &[
                (selector, ColorT::Hex("#ff0000".into())),
                (
                    ComponentSelector::Expression(row),
                    ColorT::Hex("#0000ff".into()),
                ),
            ],
            &model,
            &[0, 1, 4],
            VertexMapKind::AtomMap,
            &index,
        )
        .unwrap();
        let Some(VertexAttributeValues::Float32x4(colors)) = mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("missing colors")
        };
        assert_eq!(
            colors,
            &vec![
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0]
            ]
        );

        let source_id = ComponentSelector::Expression(ComponentExpression {
            atom_id: Some(1),
            ..Default::default()
        });
        assert!(
            evaluate_model_selector(&source_id, &model, &index)
                .err()
                .unwrap()
                .to_string()
                .contains("source atom_id")
        );
        Arc::make_mut(&mut model.hierarchy).atoms.atom_name[0] = "NEW".into();
        assert!(evaluate_model_selector(&wrong, &model, &index).is_err());
    }
}
