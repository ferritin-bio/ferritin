use super::selection::Selection;
use super::view::AtomView;
use crate::core::AtomCollection;
use pdbtbx::Element;

pub struct AtomSelector<'a> {
    collection: &'a AtomCollection,
    current_selection: Selection,
}

impl<'a> AtomSelector<'a> {
    pub(crate) fn new(collection: &AtomCollection) -> AtomSelector<'_> {
        let size = collection.get_size() as usize;
        AtomSelector {
            collection,
            current_selection: Selection::new((0..size).collect()),
        }
    }
    pub fn chain(mut self, chain_id: &str) -> Self {
        let chain_selection = self.collection.select_by_chain(chain_id);
        self.current_selection = &self.current_selection & &chain_selection;
        self
    }
    pub fn backbone(mut self) -> Self {
        unimplemented!()
    }
    pub fn collect(self) -> AtomView<'a> {
        AtomView::new(self.collection, self.current_selection)
    }

    pub fn element(mut self, element: Element) -> Self {
        let element_selection = self
            .collection
            .get_elements()
            .iter()
            .enumerate()
            .filter(|(_, &e)| e == element)
            .map(|(i, _)| i)
            .collect();
        self.current_selection = &self.current_selection & &Selection::new(element_selection);
        self
    }
    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(usize) -> bool,
    {
        let filtered = self
            .current_selection
            .indices
            .iter()
            .filter(|&&idx| predicate(idx))
            .copied()
            .collect();
        self.current_selection = Selection::new(filtered);
        self
    }
    pub fn protein(mut self) -> Self {
        unimplemented!()
    }
    pub fn residue(mut self, res_name: &str) -> Self {
        let res_selection = self.collection.select_by_residue(res_name);
        self.current_selection = &self.current_selection & &res_selection;
        self
    }

    pub fn sphere(mut self, center: [f32; 3], radius: f32) -> Self {
        let sphere_selection = self
            .collection
            .get_coords()
            .iter()
            .enumerate()
            .filter(|(_, &pos)| {
                let dx = pos[0] - center[0];
                let dy = pos[1] - center[1];
                let dz = pos[2] - center[2];
                (dx * dx + dy * dy + dz * dz).sqrt() <= radius
            })
            .map(|(i, _)| i)
            .collect();
        self.current_selection = &self.current_selection & &Selection::new(sphere_selection);
        self
    }
}
