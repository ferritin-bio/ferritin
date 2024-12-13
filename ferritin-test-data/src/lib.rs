//! ferretin-test-data
//!
//! A module to provide test files embedded in the crate for use in testing.
//! Example binary data is included in the crate distribution for reference files.
//!
//! The test files are represented as `TestFile` objects which package the raw binary data
//! and create temporary files for programs to operate on.
use std::fs;
use tempfile::{Builder, NamedTempFile};

#[derive(Debug)]
/// Test File
///
/// Example usage:
///
/// ```ignore
/// // returns (filepath, _tempfile_handle).
/// // _handle ensures the tempfile remains in scope
/// use ferritin_test_data::TestFile;
/// let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
/// let (pymol_file, _temp) = TestFile::pymol_01().create_temp().unwrap();
///
/// ```
pub struct TestFile {
    filebinary: &'static [u8],
    suffix: &'static str,
}

impl TestFile {
    pub fn protein_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/structures/101m.cif"),
            suffix: "cif",
        }
    }
    /// First Protein Use in the LigandMPNN dataset
    pub fn protein_02() -> Self {
        Self {
            filebinary: include_bytes!("../data/structures/1BC8.pdb"),
            suffix: "pdb",
        }
    }
    /// First Protein Use in the LigandMPNN dataset
    pub fn protein_03() -> Self {
        Self {
            filebinary: include_bytes!("../data/structures/1bc8.cif"),
            suffix: "cif",
        }
    }
    /// 1FAP.cif
    /// THE STRUCTURE OF THE IMMUNOPHILIN-IMMUNOSUPPRESSANT FKBP12-RAPAMYCIN COMPLEX INTERACTING WITH HUMAN FRAP
    pub fn protein_04() -> Self {
        Self {
            filebinary: include_bytes!("../data/structures/1fap.cif"),
            suffix: "cif",
        }
    }
    pub fn pymol_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/pymol/example.pse"),
            suffix: "pse",
        }
    }
    // Safetensors output of the Amplify Model
    // see `ferritin-test-data/data/safetensors/amplify/Readme.md`
    pub fn amplify_output_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/safetensors/amplify/amplify_output.safetensors"),
            suffix: "pt",
        }
    }

    /// Pytorch Model Weights for Ligand MPNN
    /// the `proteinmpnn_v_48_020.pt` file.
    pub fn ligmpnn_pmpnn_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/ligandmpnn/proteinmpnn_v_48_020.pt"),
            suffix: "pt",
        }
    }
    /// Pytorch Model Weights for Ligand MPNN
    /// the `ligandmpnn_v_32_020_25.pt` file.
    pub fn ligmpnn_lmpnn_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/ligandmpnn/ligandmpnn_v_32_020_25.pt"),
            suffix: "pt",
        }
    }
    /// Pytorch Model Weights for Ligand MPNN
    /// the `solublempnn_v_48_020.pt` file.
    pub fn ligmpnn_smpnn_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/ligandmpnn/solublempnn_v_48_020.pt"),
            suffix: "pt",
        }
    }
    /// Pytorch Model Weights for Ligand MPNN
    /// the `solublempnn_v_48_020.pt` file.
    /// ```ignore
    /// use ferritin_test_data::TestFile;
    /// let (mpnn_file, _handle) = TestFile::ligmpnn_gmpnn_01().create_temp()?;
    /// ```

    pub fn ligmpnn_gmpnn_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/ligandmpnn/global_label_membrane_mpnn_v_48_020.pt"),
            suffix: "pt",
        }
    }

    pub fn create_temp(&self) -> std::io::Result<(String, NamedTempFile)> {
        let temp = Builder::new()
            .suffix(&format!(".{}", self.suffix))
            .tempfile()?;

        fs::write(&temp, self.filebinary)?;
        let path = temp.path().to_string_lossy().into_owned();

        Ok((path, temp))
    }
}
