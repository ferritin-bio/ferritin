use super::selection::Selection;
use crate::core::AtomCollection;
use pdbtbx::Element;

pub struct AtomView<'a> {
    collection: &'a AtomCollection,
    selection: Selection,
}

impl<'a> AtomView<'a> {
    pub(crate) fn new(collection: &'a AtomCollection, selection: Selection) -> Self {
        AtomView {
            collection,
            selection,
        }
    }
    pub fn coords(&self) -> Vec<[f32; 3]> {
        self.selection
            .indices
            .iter()
            .map(|&i| *self.collection.get_coord(i))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.selection.indices.len()
    }
}
pub struct AtomRef<'a> {
    pub coords: &'a [f32; 3],
    pub res_id: &'a i32,
    pub res_name: &'a String,
    pub element: &'a Element,
    // ... other fields
}

pub struct AtomIterator<'a> {
    view: &'a AtomView<'a>,
    current: usize,
}

impl<'a> IntoIterator for &'a AtomView<'a> {
    type Item = AtomRef<'a>;
    type IntoIter = AtomIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        AtomIterator {
            view: self,
            current: 0,
        }
    }
}

impl<'a> Iterator for AtomIterator<'a> {
    type Item = AtomRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.view.selection.indices.len() {
            return None;
        }

        let idx = self.view.selection.indices[self.current];
        self.current += 1;

        // Get the actual values first
        let coords = self.view.collection.get_coord(idx);
        let res_id = self.view.collection.get_res_id(idx);
        let res_name = self.view.collection.get_res_name(idx);
        let element = self.view.collection.get_element(idx);

        Some(AtomRef {
            coords, // Pass the references directly
            res_id,
            res_name,
            element,
        })
    }
}
