use super::AtomCollection;
use std::path::PathBuf;

pub(crate) fn get_atom_container() -> AtomCollection {
    let file_path = get_file();
    let (pdb, _errors) = pdbtbx::open(file_path.to_str().unwrap()).unwrap();
    AtomCollection::from(&pdb)
}

fn get_file() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .join("tests")
        .join("data")
        .join("101m.cif")
}
