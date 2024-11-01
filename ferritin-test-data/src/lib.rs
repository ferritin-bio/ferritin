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

    pub fn to_tempfile(&self) -> std::io::Result<NamedTempFile> {
        let temp = Builder::new()
            .suffix(&format!(".{}", self.suffix))
            .tempfile()?;

        fs::write(&temp, self.filebinary)?;
        Ok(temp)
    }
}
