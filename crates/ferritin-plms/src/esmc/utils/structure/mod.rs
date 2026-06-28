//! Structure types used by the ESMC inference pipeline.
//!
//! # Ownership model
//!
//! `ferritin-core` owns the canonical protein structure (`Model` + `AtomCollection`).
//! The types in this module (`ProteinChain`, `ProteinComplex`) are **featurized
//! derivatives** — Tensor-backed representations produced on demand for ML inference.
//! They are never a source of truth; the direction of conversion is always:
//!
//!   `Model` (host, framework-agnostic) → featurize/upload → Tensor (GPU)
//!
//! The bridge lives in `ferritin-plms::featurize::StructureFeatures`, which is
//! already implemented for both `AtomCollection` and `Model`.  `candle` and
//! device/dtype concerns stay inside `ferritin-plms`; `ferritin-core` remains
//! framework-agnostic.
//!
//! `protein_chain` and `protein_complex` are gated behind this comment until the
//! ESMC SDK is ready to be wired up.  When re-enabled, `ProteinChain` must be
//! constructed via `From<&AtomCollection>` (or `featurize_atom37`), not by
//! loading directly from PDB files.

pub mod affine3d;
// pub mod protein_chain;     // WIP: enable via From<&AtomCollection>, not from_pdb
// pub mod protein_complex;   // WIP: depends on protein_chain
// pub mod protein_structure; // WIP: ProteinComplex frame/alignment utilities
