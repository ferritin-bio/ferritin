// use pymol_session_utils::molviewspec::{
//     nodes::ParseFormatT,
//     open_api::{get_builder, Metadata, Node, State, StructureParams},
// };

// #[test]
// fn test_open_api_gen() {

//     // Create StructureParams using the builder
//     let structure_params = StructureParams::builder()
//         .structure_type("model".to_string())
//         .assembly_id("1".to_string())
//         .model_index(0)
//         .build()
//         .unwrap();

//     // Create a Node using the StructureParams
//     let node = Node {
//         kind: "structure".to_string(),
//         params: Some(serde_json::to_value(structure_params).unwrap()),
//         children: None,
//     };

//     // Create a simple Metadata
//     let metadata = Metadata {
//         version: "1.0".to_string(),
//         title: Some("Example Structure".to_string()),
//         description: None,
//         description_format: None,
//         timestamp: chrono::Utc::now().to_rfc3339(),
//     };

//     // Create a State
//     let state = State {
//         root: node,
//         metadata,
//     };
// }

// #[test]
// fn test_open_api_gen2() {
//     // Create a State
//     let mut builder = get_builder();
//     let download = builder
//         .root
//         .download("https://files.rcsb.org/download/1pdb.pdb")
//         .parse("pdb")
//         .model_structure(1)
//         // .component()
//     //     .representation()
// }
