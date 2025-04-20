use ferritin_core::load_structure;
use ferritin_test_data::TestFile;

#[test]
fn test_residue_iterator() {
    let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
    let ac = load_structure(prot_file).unwrap();
    assert_eq!(ac.get_size(), 1413);
    // This includes Water Molecules
    let max_resid = ac.get_resids().iter().max().unwrap_or(&0);
    assert_eq!(*max_resid, 338);
    // this fn is only available in-crate
    // let residue_breaks = ac.get_residue_starts();
    // assert_eq!(residue_breaks, vec![1, 2, 3]);
}

#[test]
fn test_chain_iterator() {
    let (prot_file, _temp) = TestFile::protein_04().create_temp().unwrap();
    let mut ac = load_structure(prot_file).unwrap();
    ac.calculate_chain_indices();

    // Test chain iteration
    let chains: Vec<_> = ac.iter_chains().collect();
    assert_eq!(chains.len(), 4);
    assert_eq!(chains[0].chain_id(), "A");
    assert_eq!(chains[1].chain_id(), "B");
    assert_eq!(chains[2].chain_id(), "A");
    assert_eq!(chains[3].chain_id(), "B");

    // Check residue counts
    let chain_a_residue_count = chains[0].residue_count();
    let chain_b_residue_count = chains[1].residue_count();
    let chain_a1_residue_count = chains[2].residue_count();
    let chain_b1_residue_count = chains[3].residue_count();
    // assert_eq!(chain_a_residue_count, 123);
    // assert_eq!(chain_b_residue_count, 103);
    assert_eq!(
        chain_a_residue_count
            + chain_b_residue_count
            + chain_a1_residue_count
            + chain_b1_residue_count,
        ac.get_residue_start_indices().unwrap().len()
    );
}
#[test]
fn test_atom_collection_iter_residues() {
    let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
    let ac = load_structure(prot_file).unwrap();

    let residues: Vec<_> = ac.iter_residues().collect();
    assert!(!residues.is_empty());

    // Check the first residue
    let first_residue = &residues[0];
    assert!(first_residue.atom_count() > 0);

    // Check that all atoms in a residue have the same residue ID and name
    let res_id = first_residue.residue_id();
    let res_name = first_residue.residue_name();
    assert_eq!(res_id, 0);
    assert_eq!(res_name, "MET");

    // Count of atoms in all residues should match total atom count
    let total_atoms_in_residues: usize = residues.iter().map(|r| r.atom_count()).sum();
    assert_eq!(total_atoms_in_residues, ac.get_size());
}

#[test]
fn test_chain_iter_residues() {
    let (prot_file, _temp) = TestFile::protein_04().create_temp().unwrap();
    let mut ac = load_structure(prot_file).unwrap();
    ac.calculate_chain_indices();

    // Get the first chain
    let chains: Vec<_> = ac.iter_chains().collect();
    let first_chain = &chains[0];

    // Test residue iteration within a chain
    let residues: Vec<_> = first_chain.iter_residues().collect();
    assert!(!residues.is_empty());

    // All residues should be in the same chain
    let chain_id = first_chain.chain_id();
    for residue in &residues {
        assert_eq!(residue.chain_id(), chain_id);
    }
    // The number of residues should match chain's residue count
    assert_eq!(residues.len(), first_chain.residue_count());
}
