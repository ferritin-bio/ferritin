//! Combine Selections

use std::ops::BitAnd;

/// Selection
///
/// Selection are indices that can be use used to
/// identify specific sets of atoms within an [`AtomCollection`]
///
#[derive(Clone, Debug)]
pub struct Selection {
    pub(crate) indices: Vec<usize>,
}

impl Selection {
    pub fn new(indices: Vec<usize>) -> Self {
        Selection { indices }
    }

    pub fn and(&self, other: &Selection) -> Selection {
        let indices: Vec<usize> = self
            .indices
            .iter()
            .filter(|&&idx| other.indices.contains(&idx))
            .cloned()
            .collect();
        Selection::new(indices)
    }
    // Add union operation
    pub fn or(&self, other: &Selection) -> Selection {
        let mut indices = self.indices.clone();
        indices.extend(
            other
                .indices
                .iter()
                .filter(|idx| !self.indices.contains(idx)),
        );
        indices.sort();
        Selection::new(indices)
    }

    // Add difference operation
    pub fn not(&self, other: &Selection) -> Selection {
        let indices: Vec<_> = self
            .indices
            .iter()
            .filter(|&&idx| !other.indices.contains(&idx))
            .cloned()
            .collect();
        Selection::new(indices)
    }
}

impl BitAnd for &Selection {
    type Output = Selection;

    fn bitand(self, other: Self) -> Selection {
        self.and(other)
    }
}
