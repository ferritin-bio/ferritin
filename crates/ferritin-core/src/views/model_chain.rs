use super::model_residue::ModelResidueView;
use crate::model::Model;

/// Read-only view into a single chain from a [`Model`].
pub struct ModelChainView<'a> {
    pub(crate) model: &'a Model,
    pub(crate) chain_idx: usize,
}

impl<'a> ModelChainView<'a> {
    pub(crate) fn new(model: &'a Model, chain_idx: usize) -> Self {
        ModelChainView { model, chain_idx }
    }

    pub fn chain_id(&self) -> &'a str {
        &self.model.hierarchy.chains.auth_asym_id[self.chain_idx]
    }

    pub fn iter_residues(&self) -> impl Iterator<Item = ModelResidueView<'a>> + '_ {
        let range = self.model.hierarchy.residues_in_chain(self.chain_idx);
        let model = self.model;
        range.map(move |res_idx| ModelResidueView::new(model, res_idx))
    }

    pub fn residue_count(&self) -> usize {
        let range = self.model.hierarchy.residues_in_chain(self.chain_idx);
        range.end - range.start
    }
}
