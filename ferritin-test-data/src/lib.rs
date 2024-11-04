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
/// ```
/// use ferritin_test_data::TestFile;
/// let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
/// let (pymol_file, _temp) = TestFile::pymol_01().create_temp().unwrap();
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

    pub fn pymol_01() -> Self {
        Self {
            filebinary: include_bytes!("../data/pymol/example.pse"),
            suffix: "pse",
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
