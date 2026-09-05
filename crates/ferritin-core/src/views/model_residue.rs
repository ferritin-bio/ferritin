use super::model_atom::ModelAtomView;
use crate::info::constants::{is_amino_acid, is_carbohydrate, is_nucleotide};
use crate::model::Model;

/// Read-only view into a single residue from a [`Model`].
pub struct ModelResidueView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) res_idx: usize,
}

impl<'a> ModelResidueView<'a> {
    pub(crate) fn new(model: &'a Model, res_idx: usize) -> Self {
        ModelResidueView { model, res_idx }
    }

    pub fn atom_count(&self) -> usize {
        let range = self.model.hierarchy.atoms_in_residue(self.res_idx);
        range.end - range.start
    }

    pub fn chain_id(&self) -> &'a str {
        let chain_idx = self.model.hierarchy.chain_of_residue(self.res_idx);
        &self.model.hierarchy.chains.auth_asym_id[chain_idx]
    }

    /// Canonical chain identifier (`label_asym_id`).
    pub fn label_chain_id(&self) -> &'a str {
        let chain_idx = self.model.hierarchy.chain_of_residue(self.res_idx);
        &self.model.hierarchy.chains.label_asym_id[chain_idx]
    }

    pub fn is_amino_acid(&self) -> bool {
        is_amino_acid(self.residue_name())
    }

    pub fn is_nucleotide(&self) -> bool {
        is_nucleotide(self.residue_name())
    }

    pub fn is_carbohydrate(&self) -> bool {
        is_carbohydrate(self.residue_name())
    }

    pub fn iter_atoms(&self) -> impl Iterator<Item = ModelAtomView<'a>> + '_ {
        let range = self.model.hierarchy.atoms_in_residue(self.res_idx);
        let model = self.model;
        range.map(move |atom_idx| ModelAtomView::new(model, atom_idx))
    }

    pub fn residue_id(&self) -> i32 {
        self.model.hierarchy.residues.auth_seq_id[self.res_idx]
    }

    pub fn residue_name(&self) -> &'a str {
        &self.model.hierarchy.residues.comp_id[self.res_idx]
    }

    /// Author-assigned component identifier (`auth_comp_id`).
    pub fn auth_residue_name(&self) -> &'a str {
        &self.model.hierarchy.residues.auth_comp_id[self.res_idx]
    }

    pub fn find_atom_by_name(&self, name: &str) -> Option<ModelAtomView<'a>> {
        let mut range = self.model.hierarchy.atoms_in_residue(self.res_idx);
        let model = self.model;
        range
            .find(|&i| model.hierarchy.atoms.atom_name[i] == name)
            .map(|idx| ModelAtomView::new(model, idx))
    }
}
