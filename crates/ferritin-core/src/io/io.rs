use crate::AtomCollection;
use crate::io::cif;
use crate::io::pdb;
use crate::model::Model;
use crate::trajectory::ArrayTrajectory;
use anyhow::{Context, Result};
use std::path::Path;

//
pub fn load_structure<P: AsRef<Path>>(file_path: P) -> Result<AtomCollection> {
    let path = file_path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| anyhow::anyhow!("File has no extension"))?
        .to_lowercase();

    let mut ac = match extension.as_str() {
        "pdb" => pdb::PDBFile::read(path)
            .context("Failed to read PDB file")?
            .parse_to_atom_collection()
            .context("Failed to parse PDB file to atom collection")?,
        "cif" => cif::CIFFile::read(path)?.parse_to_atom_collection()?,
        _ => return Err(anyhow::anyhow!("Unsupported file extension: {}", extension)),
    };

    ac.connect_via_residue_names();
    Ok(ac)
}

/// Load all models from a structure file as a trajectory.
///
/// For single-model files, returns a trajectory with one frame.
/// For multi-model NMR/MD files, returns all frames sharing one `Arc<AtomicHierarchy>`.
pub fn load_trajectory<P: AsRef<Path>>(file_path: P) -> Result<ArrayTrajectory> {
    let path = file_path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| anyhow::anyhow!("File has no extension"))?
        .to_lowercase();

    match extension.as_str() {
        "cif" => cif::CIFFile::read(path)?
            .parse_to_trajectory()
            .context("Failed to parse CIF file to trajectory"),
        "pdb" => Err(anyhow::anyhow!(
            "PDB multi-model trajectory not yet implemented"
        )),
        _ => Err(anyhow::anyhow!("Unsupported file extension: {}", extension)),
    }
}

/// Load the representative (first) model from a structure file as a [`Model`].
///
/// For multi-model files, only the first model is returned. Use [`load_trajectory`]
/// to access all frames.
///
/// Currently supported: `.cif`. PDB single-model support pending.
pub fn load_model<P: AsRef<Path>>(file_path: P) -> Result<Model> {
    let path = file_path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| anyhow::anyhow!("File has no extension"))?
        .to_lowercase();

    match extension.as_str() {
        "cif" => cif::CIFFile::read(path)?
            .parse_to_model()
            .context("Failed to parse CIF file to model"),
        "pdb" => Err(anyhow::anyhow!(
            "load_model for PDB not yet implemented; use load_trajectory for CIF files"
        )),
        _ => Err(anyhow::anyhow!("Unsupported file extension: {}", extension)),
    }
}

pub fn load_structure_from_string(content: &str, filetype: &str) -> Result<AtomCollection> {
    let filetype = filetype.to_lowercase();

    let mut ac = match filetype.as_str() {
        "pdb" => pdb::PDBFile::new_from_string(content.to_string())
            .context("Failed to read PDB from string")?
            .parse_to_atom_collection()
            .context("Failed to parse PDB string to atom collection")?,
        "cif" => cif::CIFFile::new(content.to_string())
            .context("Failed to read CIF from string")?
            .parse_to_atom_collection()
            .context("Failed to parse CIF string to atom collection")?,
        _ => return Err(anyhow::anyhow!("Unsupported file type: {}", filetype)),
    };

    ac.connect_via_residue_names();
    Ok(ac)
}
