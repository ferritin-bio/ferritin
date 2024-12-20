// ferritin-core/src/features/ndarray_impl.rs
use super::{ProteinFeatures, StructureFeatures};
use crate::AtomCollection;
use ndarray::{Array, Array2};

impl StructureFeatures<Array2<f32>> for AtomCollection {
    type Error = ndarray::ShapeError;

    fn encode_amino_acids(&self) -> Result<Array2<f32>, Self::Error> {
        let n = self.iter_residues_aminoacid().count();
        let s: Vec<f32> = self
            .iter_residues_aminoacid()
            .map(|res| res.res_name)
            .map(|res| aa3to1(&res))
            .map(|res| aa1to_int(res) as f32)
            .collect();

        Array2::from_shape_vec((1, n), s)
    }

    // ... implement other methods
}

impl ProteinFeatures<Array2<f32>> {
    pub fn save_to_npy(&self, path: &str) -> Result<(), std::io::Error> {
        // Implementation for saving to .npy format
        todo!()
    }
}
