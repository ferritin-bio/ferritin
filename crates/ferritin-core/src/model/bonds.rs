//! Bond connectivity stored in struct-of-arrays (SoA) layout with CSR indexing.
//!
//! After construction via [`Bonds::from_unsorted`], bonds are sorted by `atom_a`
//! and `atom_bond_starts` provides O(1) access to all bonds for a given atom.

/// Bond connectivity stored as SoA (struct-of-arrays).
///
/// `atom_a` and `atom_b` are atom indices; `order` is bond order (1 = single, 2 = double, etc.).
///
/// After construction the arrays are sorted by `atom_a` so that
/// `atom_bond_starts[i]..atom_bond_starts[i+1]` gives the bond range for atom `i`.
#[derive(Clone, Debug)]
pub struct Bonds {
    /// First atom index for each bond (sorted).
    pub atom_a: Vec<u32>,
    /// Second atom index for each bond.
    pub atom_b: Vec<u32>,
    /// Bond order for each bond (1 = single, 2 = double, 3 = triple, etc.).
    pub order: Vec<u8>,
    /// CSR-style start index into bonds arrays, indexed by atom.
    ///
    /// `atom_bond_starts.len() == n_atoms + 1`; atom `i`'s bonds span
    /// `atom_bond_starts[i]..atom_bond_starts[i+1]`.
    pub atom_bond_starts: Vec<u32>,
}

impl Bonds {
    /// Construct from parallel atom_a/atom_b/order vectors.
    ///
    /// Sorts bonds by `atom_a` and builds the `atom_bond_starts` CSR index.
    ///
    /// # Panics
    /// Panics if `atom_a`, `atom_b`, and `order` have different lengths,
    /// or if any atom index is >= `n_atoms`.
    pub fn from_unsorted(
        atom_a: Vec<u32>,
        atom_b: Vec<u32>,
        order: Vec<u8>,
        n_atoms: usize,
    ) -> Self {
        assert_eq!(atom_a.len(), atom_b.len(), "atom_a and atom_b must have the same length");
        assert_eq!(atom_a.len(), order.len(), "atom_a and order must have the same length");

        let n_bonds = atom_a.len();

        // Build sort permutation by atom_a
        let mut indices: Vec<usize> = (0..n_bonds).collect();
        indices.sort_by_key(|&i| atom_a[i]);

        let sorted_a: Vec<u32> = indices.iter().map(|&i| atom_a[i]).collect();
        let sorted_b: Vec<u32> = indices.iter().map(|&i| atom_b[i]).collect();
        let sorted_order: Vec<u8> = indices.iter().map(|&i| order[i]).collect();

        // Build CSR atom_bond_starts: atom_bond_starts[i] = first bond index for atom i
        let mut atom_bond_starts = vec![0u32; n_atoms + 1];
        for &a in &sorted_a {
            let a = a as usize;
            assert!(a < n_atoms, "atom index {} out of range (n_atoms={})", a, n_atoms);
            atom_bond_starts[a + 1] += 1;
        }
        // prefix-sum
        for i in 1..=n_atoms {
            atom_bond_starts[i] += atom_bond_starts[i - 1];
        }

        Self {
            atom_a: sorted_a,
            atom_b: sorted_b,
            order: sorted_order,
            atom_bond_starts,
        }
    }

    /// Returns an iterator over `(atom_b, order)` pairs for all bonds from `atom_idx`.
    ///
    /// Only bonds where `atom_idx` appears as `atom_a` are returned. For undirected
    /// traversal, callers should also check the reverse (atom_b side) or store both
    /// directions when constructing.
    pub fn bonds_for_atom(&self, atom_idx: usize) -> impl Iterator<Item = (u32, u8)> + '_ {
        let start = self.atom_bond_starts[atom_idx] as usize;
        let end = self.atom_bond_starts[atom_idx + 1] as usize;
        self.atom_b[start..end]
            .iter()
            .zip(self.order[start..end].iter())
            .map(|(&b, &o)| (b, o))
    }

    /// Number of bonds.
    pub fn len(&self) -> usize {
        self.atom_a.len()
    }

    /// Returns `true` if there are no bonds.
    pub fn is_empty(&self) -> bool {
        self.atom_a.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonds_from_unsorted_sorted_correctly() {
        // Provide bonds out of order: (2->3), (0->1), (1->2)
        let atom_a = vec![2u32, 0, 1];
        let atom_b = vec![3u32, 1, 2];
        let order = vec![1u8, 2, 1];

        let bonds = Bonds::from_unsorted(atom_a, atom_b, order, 4);

        // After sorting by atom_a the order should be: (0->1,2), (1->2,1), (2->3,1)
        assert_eq!(bonds.atom_a, vec![0, 1, 2]);
        assert_eq!(bonds.atom_b, vec![1, 2, 3]);
        assert_eq!(bonds.order, vec![2, 1, 1]);

        // CSR index: atom_bond_starts should be [0, 1, 2, 3, 3]
        assert_eq!(bonds.atom_bond_starts, vec![0, 1, 2, 3, 3]);
        assert_eq!(bonds.len(), 3);
    }

    #[test]
    fn test_bonds_for_atom() {
        let atom_a = vec![0u32, 0, 1];
        let atom_b = vec![1u32, 2, 2];
        let order = vec![1u8, 2, 1];

        let bonds = Bonds::from_unsorted(atom_a, atom_b, order, 3);

        // Atom 0 has bonds to 1 (order=1) and 2 (order=2)
        let bonds_0: Vec<(u32, u8)> = bonds.bonds_for_atom(0).collect();
        assert_eq!(bonds_0.len(), 2);
        assert!(bonds_0.contains(&(1, 1)));
        assert!(bonds_0.contains(&(2, 2)));

        // Atom 1 has a bond to 2 (order=1)
        let bonds_1: Vec<(u32, u8)> = bonds.bonds_for_atom(1).collect();
        assert_eq!(bonds_1, vec![(2, 1)]);

        // Atom 2 has no outgoing bonds (only incoming)
        let bonds_2: Vec<(u32, u8)> = bonds.bonds_for_atom(2).collect();
        assert!(bonds_2.is_empty());
    }

    #[test]
    fn test_bonds_empty() {
        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], 5);
        assert!(bonds.is_empty());
        assert_eq!(bonds.len(), 0);
        assert_eq!(bonds.atom_bond_starts, vec![0, 0, 0, 0, 0, 0]);
    }
}
