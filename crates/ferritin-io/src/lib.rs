use anyhow::{Context, Result};
use ferritin_core::AtomCollection;
use std::path::Path;
mod cif;
mod pdb;

//
fn load_structure<P: AsRef<Path>>(file_path: P) -> Result<AtomCollection> {
    let path = file_path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| anyhow::anyhow!("File has no extension"))?
        .to_lowercase();

    match extension.as_str() {
        "pdb" => Ok(pdb::PDBFile::read(path)
            .context("Failed to read PDB file")?
            .parse_to_atom_collection()
            .context("Failed to parse PDB file to atom collection")?),
        "cif" => Ok(cif::CIFFile::read(path)?.parse_to_atom_collection()?),
        _ => Err(anyhow::anyhow!("Unsupported file extension: {}", extension)),
    }
}
