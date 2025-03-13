/// View representing a residue (amino acid, nucleotide, etc.) in the molecule.
pub struct ResidueView<'a> {
    data: &'a MoleculeData,
    start_atom_idx: usize,
    end_atom_idx: usize,
    residue_idx: usize,
}
