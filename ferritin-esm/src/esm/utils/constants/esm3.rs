// use cached::proc_macro::cached;
// use huggingface_hub::snapshot_download;
// use std::env;
// use std::path::PathBuf;

const SEQUENCE_BOS_TOKEN: i32 = 0;
const SEQUENCE_PAD_TOKEN: i32 = 1;
const SEQUENCE_EOS_TOKEN: i32 = 2;
const SEQUENCE_CHAINBREAK_TOKEN: i32 = 31;
const SEQUENCE_MASK_TOKEN: i32 = 32;

const VQVAE_CODEBOOK_SIZE: i32 = 4096;

// lazy_static! {
//     static ref VQVAE_SPECIAL_TOKENS: std::collections::HashMap<&'static str, i32> = {
//         let mut m = std::collections::HashMap::new();
//         m.insert("MASK", VQVAE_CODEBOOK_SIZE);
//         m.insert("EOS", VQVAE_CODEBOOK_SIZE + 1);
//         m.insert("BOS", VQVAE_CODEBOOK_SIZE + 2);
//         m.insert("PAD", VQVAE_CODEBOOK_SIZE + 3);
//         m.insert("CHAINBREAK", VQVAE_CODEBOOK_SIZE + 4);
//         m
//     };
// }
// const VQVAE_DIRECTION_LOSS_BINS: i32 = 16;
// const VQVAE_PAE_BINS: i32 = 64;
// const VQVAE_MAX_PAE_BIN: f32 = 31.0;
// const VQVAE_PLDDT_BINS: i32 = 50;

const STRUCTURE_MASK_TOKEN: i32 = VQVAE_CODEBOOK_SIZE;
const STRUCTURE_BOS_TOKEN: i32 = VQVAE_CODEBOOK_SIZE + 2;
const STRUCTURE_EOS_TOKEN: i32 = VQVAE_CODEBOOK_SIZE + 1;
const STRUCTURE_PAD_TOKEN: i32 = VQVAE_CODEBOOK_SIZE + 3;
const STRUCTURE_CHAINBREAK_TOKEN: i32 = VQVAE_CODEBOOK_SIZE + 4;
const STRUCTURE_UNDEFINED_TOKEN: i32 = 955;

const SASA_PAD_TOKEN: i32 = 0;

const SS8_PAD_TOKEN: i32 = 0;

const INTERPRO_PAD_TOKEN: i32 = 0;

const RESIDUE_PAD_TOKEN: i32 = 0;

const CHAIN_BREAK_STR: &str = "|";

const SEQUENCE_BOS_STR: &str = "<cls>";
const SEQUENCE_EOS_STR: &str = "<eos>";

pub const MASK_STR_SHORT: &str = "_";
const SEQUENCE_MASK_STR: &str = "<mask>";
const SASA_MASK_STR: &str = "<unk>";
const SS8_MASK_STR: &str = "<unk>";

pub const SEQUENCE_VOCAB: &[&str] = &[
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
];

const SSE_8CLASS_VOCAB: &str = "GHITEBSC";
const SSE_3CLASS_VOCAB: &str = "HEC";

// lazy_static! {
//     static ref SSE_8CLASS_TO_3CLASS_MAP: std::collections::HashMap<&'static str, &'static str> = {
//         let mut m = std::collections::HashMap::new();
//         m.insert("G", "H");
//         m.insert("H", "H");
//         m.insert("I", "H");
//         m.insert("T", "C");
//         m.insert("E", "E");
//         m.insert("B", "E");
//         m.insert("S", "C");
//         m.insert("C", "C");
//         m
//     };
// }

const SASA_DISCRETIZATION_BOUNDARIES: &[f32] = &[
    0.8, 4.0, 9.6, 16.4, 24.5, 32.9, 42.0, 51.5, 61.2, 70.9, 81.6, 93.3, 107.2, 125.4, 151.4,
];

const MAX_RESIDUE_ANNOTATIONS: i32 = 16;

const TFIDF_VECTOR_SIZE: i32 = 58641;

// #[cached]
// fn data_root(model: &str) -> PathBuf {
//     if env::var("INFRA_PROVIDER").is_ok() {
//         return PathBuf::from("");
//     }

//     let path = match model {
//         m if m.starts_with("esm3") => {
//             snapshot_download("EvolutionaryScale/esm3-sm-open-v1").unwrap()
//         }
//         m if m.starts_with("esmc-300") => {
//             snapshot_download("EvolutionaryScale/esmc-300m-2024-12").unwrap()
//         }
//         m if m.starts_with("esmc-600") => {
//             snapshot_download("EvolutionaryScale/esmc-600m-2024-12").unwrap()
//         }
//         _ => panic!("{:?} is an invalid model name", model),
//     };

//     PathBuf::from(path)
// }

// lazy_static! {
//     static ref IN_REPO_DATA_FOLDER: PathBuf =
//         PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
// }

// lazy_static! {
//     static ref INTERPRO_ENTRY: PathBuf = IN_REPO_DATA_FOLDER.join("entry_list_safety_29026.list");
//     static ref INTERPRO_HIERARCHY: PathBuf = IN_REPO_DATA_FOLDER.join("ParentChildTreeFile.txt");
//     static ref INTERPRO2GO: PathBuf = IN_REPO_DATA_FOLDER.join("ParentChildTreeFile.txt");
// }

// const INTERPRO_2ID: &str = "data/tag_dict_4_safety_filtered.json";

// lazy_static! {
//     static ref LSH_TABLE_PATHS: std::collections::HashMap<&'static str, &'static str> = {
//         let mut m = std::collections::HashMap::new();
//         m.insert("8bit", "data/hyperplanes_8bit_58641.npz");
//         m
//     };
// }

// lazy_static! {
//     static ref KEYWORDS_VOCABULARY: PathBuf =
//         IN_REPO_DATA_FOLDER.join("keyword_vocabulary_safety_filtered_58641.txt");
//     static ref KEYWORDS_IDF: PathBuf =
//         IN_REPO_DATA_FOLDER.join("keyword_idf_safety_filtered_58641.npy");
// }

// const RESID_CSV: &str = "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv";

// lazy_static! {
//     static ref INTERPRO2KEYWORDS: PathBuf =
//         IN_REPO_DATA_FOLDER.join("interpro_29026_to_keywords_58641.csv");
// }
