use std::fs;
use tempfile::{Builder, NamedTempFile};

#[derive(Debug)]
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
