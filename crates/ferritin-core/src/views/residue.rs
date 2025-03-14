use crate::AtomCollection;

/// View representing a residue (amino acid, nucleotide, etc.) in the molecule.
pub struct ResidueView<'a> {
    pub(crate) data: &'a AtomCollection,
    pub(crate) start_atom_idx: usize,
    pub(crate) end_atom_idx: usize,
}

impl<'a> ResidueView<'a> {
    pub fn new(data: &'a AtomCollection, start_atom_idx: usize, end_atom_idx: usize) -> Self {
        ResidueView {
            data,
            start_atom_idx,
            end_atom_idx,
        }
    }

    // pub fn atom_count(&self) -> usize {
    //     self.end_atom_idx - self.start_atom_idx
    // }

    // pub fn residue_name(&self) -> &str {
    //     &self.data.get_res_name(self.start_atom_idx)
    // }

    // pub fn residue_id(&self) -> i32 {
    //     *self.data.get_res_id(self.start_atom_idx)
    // }

    // pub fn chain_id(&self) -> &str {
    //     self.data.get_chain_id(self.start_atom_idx)
    // }

    // pub fn is_amino_acid(&self) -> bool {
    //     // Standard amino acid check
    //     let name = self.residue_name();
    //     matches!(
    //         name,
    //         "ALA"
    //             | "ARG"
    //             | "ASN"
    //             | "ASP"
    //             | "CYS"
    //             | "GLN"
    //             | "GLU"
    //             | "GLY"
    //             | "HIS"
    //             | "ILE"
    //             | "LEU"
    //             | "LYS"
    //             | "MET"
    //             | "PHE"
    //             | "PRO"
    //             | "SER"
    //             | "THR"
    //             | "TRP"
    //             | "TYR"
    //             | "VAL"
    //     )
    // }

    // pub fn is_hetero(&self) -> bool {
    //     self.data.get_is_hetero(self.start_atom_idx)
    // }

    // pub fn atoms(&self) -> Selection {
    //     let indices: Vec<usize> = (self.start_atom_idx..self.end_atom_idx).collect();
    //     Selection::new(indices)
    // }

    // // Get atoms by name within this residue
    // pub fn get_atom_by_name(&self, name: &str) -> Option<usize> {
    //     (self.start_atom_idx..self.end_atom_idx).find(|&i| self.data.get_atom_name(i) == name)
    // }
}
