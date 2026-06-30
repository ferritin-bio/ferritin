use crate::info::elements::Element;
use crate::model::Model;

/// Read-only view into a single atom from a [`Model`].
pub struct ModelAtomView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) idx: usize,
}

impl<'a> ModelAtomView<'a> {
    pub(crate) fn new(model: &'a Model, idx: usize) -> Self {
        ModelAtomView { model, idx }
    }

    /// `[x, y, z]` for this atom (AoS convenience; allocation-free from SoA).
    pub fn coords(&self) -> [f32; 3] {
        self.model.coord(self.idx)
    }

    pub fn element(&self) -> Element {
        self.model.hierarchy.atoms.element[self.idx]
    }

    pub fn atom_name(&self) -> &'a str {
        &self.model.hierarchy.atoms.atom_name[self.idx]
    }

    pub fn residue_id(&self) -> i32 {
        let res_idx = self.model.hierarchy.residue_of_atom(self.idx);
        self.model.hierarchy.residues.auth_seq_id[res_idx]
    }

    pub fn residue_name(&self) -> &'a str {
        let res_idx = self.model.hierarchy.residue_of_atom(self.idx);
        &self.model.hierarchy.residues.comp_id[res_idx]
    }

    pub fn chain_id(&self) -> &'a str {
        let res_idx = self.model.hierarchy.residue_of_atom(self.idx);
        let chain_idx = self.model.hierarchy.chain_of_residue(res_idx);
        &self.model.hierarchy.chains.auth_asym_id[chain_idx]
    }

    pub fn is_hetero(&self) -> bool {
        let res_idx = self.model.hierarchy.residue_of_atom(self.idx);
        self.model.hierarchy.residues.group[res_idx]
            == crate::model::tables::ResidueGroup::NonPolymer
    }

    pub fn index(&self) -> usize {
        self.idx
    }
}
