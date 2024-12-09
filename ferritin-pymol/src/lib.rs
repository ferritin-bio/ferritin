//! # ferritin-pymol
//!
//! A Rust crate for working with PyMOL PSE (PyMOL Session) files.
//!
//! ## Features
//!
//! - Load and parse PSE files
//! - Serialize PSE data
//! - Access molecular structures and visualization settings
//!
//! ## Usage
//!
//! ```no_run
//! use ferritin_pymol::pymolparsing::psedata::PSEData;
//! let psedata = PSEData::load("path/to/file.pse").expect("local pse path");
//! // Work with the loaded PSE data
//! psedata.to_disk_full("my_output_directory");
//! ```
//!
pub mod pymolparsing;
pub mod conversions;
pub use self::pymolparsing::parsing::PyObjectMolecule;
pub use self::pymolparsing::psedata::PSEData;
