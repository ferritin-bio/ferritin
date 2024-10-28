use super::selection::Selection;
use crate::core::AtomCollection;
use pdbtbx::Element;

pub struct AtomView<'a> {
    collection: &'a AtomCollection,
    selection: Selection,
}

impl<'a> AtomView<'a> {
    pub(crate) fn new(collection: &'a AtomCollection, selection: Selection) -> Self {
        Self {
            collection,
            selection,
        }
    }

    pub fn coords(&self) -> Vec<[f32; 3]> {
        self.selection
            .indices
            .iter()
            .map(|&i| self.collection.coords()[i])
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

        Some(AtomRef {
            coords: &self.view.collection.coords()[idx],
            res_id: &self.view.collection.resids()[idx],
            res_name: &self.view.collection.resnames()[idx],
            element: &self.view.collection.elements()[idx],
        })
    }
}
