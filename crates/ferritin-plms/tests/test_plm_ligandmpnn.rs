// //! Test the LigandMPNN models
// //!
// //! This test file validates loading and running the LigandMPNN models,
// //! which are used for protein-ligand interaction prediction.

// use candle_core::Device;
// use ferritin_plms::ligandmpnn::ligandmpnn_runner::{LigandMpnnModels, LigandMpnnRunner};
// use ferritin_plms::device;

// // Test protein sequence
// const TEST_SEQUENCE: &str = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";

// // Test ligand SMILES string
// const TEST_LIGAND: &str = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5";

// /// Test that we can successfully load the LigandMPNN model
// #[test]
// #[ignore = "requires downloading model files"]
// fn test_load_ligandmpnn() {
//     let ligandmpnn = LigandMpnnRunner::load_model(LigandMpnnModels::BASE, device().unwrap());
//     assert!(
//         ligandmpnn.is_ok(),
//         "Failed to load LigandMPNN model: {:?}",
//         ligandmpnn.err()
//     );
// }

// /// Test binding site prediction using LigandMPNN
// #[test]
// #[ignore = "requires downloading model files"]
// fn test_ligandmpnn_binding_site() {
//     let ligandmpnn = LigandMpnnRunner::load_model(LigandMpnnModels::BASE, device().unwrap())
//         .expect("Failed to load LigandMPNN model");
//     let sequence = TEST_SEQUENCE;
//     let ligand = TEST_LIGAND;

//     // Predict binding site residues
//     let binding_sites = ligandmpnn.predict_binding_sites(sequence, ligand);
//     assert!(
//         binding_sites.is_ok(),
//         "Failed to predict binding sites: {:?}",
//         binding_sites.err()
//     );

//     let sites = binding_sites.unwrap();
//     assert!(!sites.is_empty(), "Binding site predictions should not be empty");
//     println!("Predicted {} binding site residues", sites.len());

//     // Basic validation of binding site predictions
//     for site in &sites {
//         assert!(site.position < sequence.len(), "Position should be valid");
//         assert!(
//             !site.score.is_nan(),
//             "Binding score should be a valid number"
//         );
//     }
// }

// /// Test affinity prediction using LigandMPNN
// #[test]
// #[ignore = "requires downloading model files"]
// fn test_ligandmpnn_affinity() {
//     let ligandmpnn = LigandMpnnRunner::load_model(LigandMpnnModels::BASE, device().unwrap())
//         .expect("Failed to load LigandMPNN model");
//     let sequence = TEST_SEQUENCE;
//     let ligand = TEST_LIGAND;

//     // Predict binding affinity
//     let affinity = ligandmpnn.predict_affinity(sequence, ligand);
//     assert!(
//         affinity.is_ok(),
//         "Failed to predict affinity: {:?}",
//         affinity.err()
//     );

//     let affinity_score = affinity.unwrap();
//     assert!(
//         !affinity_score.is_nan(),
//         "Affinity score should be a valid number"
//     );
//     println!("Predicted binding affinity: {}", affinity_score);
// }

// /// Test mutation effect prediction using LigandMPNN
// #[test]
// #[ignore = "requires downloading model files"]
// fn test_ligandmpnn_mutation_effect() {
//     let ligandmpnn = LigandMpnnRunner::load_model(LigandMpnnModels::BASE, device().unwrap())
//         .expect("Failed to load LigandMPNN model");
//     let sequence = TEST_SEQUENCE;
//     let ligand = TEST_LIGAND;

//     // Define some test mutations (position, wild type, mutant)
//     let mutations = vec![
//         (10, 'L', 'A'),  // Leucine to Alanine at position 10
//         (25, 'K', 'R'),  // Lysine to Arginine at position 25
//     ];

//     // Predict effect of mutations on binding
//     let mutation_effects = ligandmpnn.predict_mutation_effects(sequence, ligand, &mutations);
//     assert!(
//         mutation_effects.is_ok(),
//         "Failed to predict mutation effects: {:?}",
//         mutation_effects.err()
//     );

//     let effects = mutation_effects.unwrap();
//     assert_eq!(
//         effects.len(),
//         mutations.len(),
//         "Should have effect predictions for each mutation"
//     );

//     // Validate mutation effect predictions
//     for (i, effect) in effects.iter().enumerate() {
//         let (pos, wt, mt) = mutations[i];
//         assert_eq!(effect.position, pos, "Position should match");
//         assert_eq!(effect.wild_type, wt, "Wild type should match");
//         assert_eq!(effect.mutant, mt, "Mutant should match");
//         assert!(
//             !effect.delta_affinity.is_nan(),
//             "Delta affinity should be a valid number"
//         );
//         println!("Mutation {}{}{}: ΔΔG = {}", wt, pos, mt, effect.delta_affinity);
//     }
// }
