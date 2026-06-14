//! ESM3 pretrained weight loading (stub).
//!
//! Full implementation tracked by ferritin-fdz.
//! Weights are gated-access on HuggingFace:
//!   <https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1>

/// Available ESM3 model variants.
pub enum ESM3Models {
    /// esm3-sm-open-v1 — 1.4B parameter open-access model.
    SmOpen,
}

impl ESM3Models {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::SmOpen => "EvolutionaryScale/esm3-sm-open-v1",
        }
    }
}
