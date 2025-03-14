/// View representing a single atom in the molecule.
pub struct AtomView<'a> {
    data: &'a MoleculeData,
    atom_idx: usize,
}
