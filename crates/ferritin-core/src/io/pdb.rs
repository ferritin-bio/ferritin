//! Low-level PDB file parsing and writing.
//!
//! Adapted from: https://github.com/biotite-dev/fastpdb/tree/main
//! Converted to use native Rust structures instead of Python/NumPy bindings.
//!
use crate::info::elements::Element;
use crate::{AtomCollection, Bond, BondOrder};
use anyhow::Result;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::str::FromStr;

/// Custom error types for PDB parsing operations
#[derive(Debug)]
pub enum PDBError {
    InvalidFile(String),
    IOError(std::io::Error),
    #[allow(dead_code)]
    ValueError(String),
    OSError(String),
}

impl fmt::Display for PDBError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PDBError::InvalidFile(msg) => write!(f, "Invalid PDB file: {}", msg),
            PDBError::IOError(err) => write!(f, "IO error: {}", err),
            PDBError::ValueError(msg) => write!(f, "Value error: {}", msg),
            PDBError::OSError(msg) => write!(f, "OS error: {}", msg),
        }
    }
}

impl Error for PDBError {}

impl From<std::io::Error> for PDBError {
    fn from(err: std::io::Error) -> Self {
        PDBError::IOError(err)
    }
}

/// This is a low-level abstraction of a PDB file.
/// The struct is able to parse coordinates, models, bonds etc. from lines of text and vice versa.
pub struct PDBFile {
    /// Lines of text from the PDB file.
    pub lines: Vec<String>,
    model_start_i: Vec<usize>,
    atom_line_i: Vec<usize>,
}

impl PDBFile {
    /// Create a new [`PDBFile`].
    /// The lines of text are given to `lines`.
    /// An empty `Vec` represents an empty PDB file.
    pub fn new(lines: Vec<String>) -> Result<PDBFile> {
        let mut pdb_file = PDBFile {
            lines,
            model_start_i: Vec::new(),
            atom_line_i: Vec::new(),
        };
        pdb_file.index_models_and_atoms();
        Ok(pdb_file)
    }

    /// Create a new [`PDBFile`] from the full text of the file.
    pub fn new_from_string(content: String) -> Result<PDBFile> {
        let lines = content.lines().map(|x| x.to_string()).collect();
        PDBFile::new(lines)
    }

    /// Read a [`PDBFile`] from a file.
    /// The file is indicated by its file path as `String`.
    pub fn read<P: AsRef<std::path::Path>>(file_path: P) -> Result<Self, PDBError> {
        let path = file_path.as_ref();
        let contents = fs::read_to_string(path)
            .map_err(|_| PDBError::OSError(format!("'{}' cannot be read", path.display())))?;
        let lines = contents
            .lines()
            .map(|line| format!("{:<80}", line))
            .collect();
        let mut pdb_file = PDBFile {
            lines,
            model_start_i: Vec::new(),
            atom_line_i: Vec::new(),
        };
        pdb_file.index_models_and_atoms();
        Ok(pdb_file)
    }

    /// Get indices to the start positions of all models in the file.
    #[allow(dead_code)]
    pub fn get_model_start_indices(&self) -> Vec<usize> {
        self.model_start_i.clone()
    }

    /// Get indices to all `ATOM` and `HETATM` records in the file.
    #[allow(dead_code)]
    pub fn get_atom_line_indices(&self) -> Vec<usize> {
        self.atom_line_i.clone()
    }

    /// Get the number of models contained in the file.
    #[allow(dead_code)]
    pub fn get_model_count(&self) -> usize {
        self.model_start_i.len()
    }

    /// Parse the given `REMARK` record of the PDB file to obtain its content as strings
    #[allow(dead_code)]
    pub fn parse_remark(&self, number: i64) -> Result<Option<Vec<String>>, PDBError> {
        const CONTENT_START_COLUMN: usize = 11;

        if !(0..=999).contains(&number) {
            return Err(PDBError::ValueError(
                "The number must be in range 0-999".to_string(),
            ));
        }
        let remark_string = format!("REMARK {:>3}", number);
        let mut remark_lines: Vec<String> = self
            .lines
            .iter()
            .filter(|line| line.starts_with(&remark_string))
            .map(|line| line[CONTENT_START_COLUMN..].to_owned())
            .collect();
        match remark_lines.len() {
            0 => Ok(None),
            // Remove first empty line
            _ => {
                remark_lines.remove(0);
                Ok(Some(remark_lines))
            }
        }
    }

    /// Parse the `CRYST1` record of the PDB file to obtain the unit cell lengths
    /// and angles (in degrees).
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)] // (a, b, c, alpha, beta, gamma) unit-cell tuple
    pub fn parse_box(&self) -> Result<Option<(f32, f32, f32, f32, f32, f32)>, PDBError> {
        for line in self.lines.iter() {
            if line.starts_with("CRYST1") {
                if line.len() < 80 {
                    return Err(PDBError::InvalidFile("Line is too short".to_string()));
                }
                let len_a = parse_number(&line[6..15])?;
                let len_b = parse_number(&line[15..24])?;
                let len_c = parse_number(&line[24..33])?;
                let alpha = parse_number(&line[33..40])?;
                let beta = parse_number(&line[40..47])?;
                let gamma = parse_number(&line[47..54])?;
                return Ok(Some((len_a, len_b, len_c, alpha, beta, gamma)));
            }
        }
        // File has no 'CRYST1' record
        Ok(None)
    }

    /// Parse PDB file into an AtomCollection
    pub fn parse_to_atom_collection(
        &self,
        // model: Option<isize>,
    ) -> Result<AtomCollection, PDBError> {
        let atom_line_i = self.atom_line_i.clone();
        let size = atom_line_i.len();
        let mut coords = Vec::with_capacity(size);
        let mut res_ids = Vec::with_capacity(size);
        let mut res_names = Vec::with_capacity(size);
        let mut is_hetero = Vec::with_capacity(size);
        let mut elements = Vec::with_capacity(size);
        let mut atom_names = Vec::with_capacity(size);
        let mut chain_ids = Vec::with_capacity(size);
        let mut atom_ids = Vec::with_capacity(size);

        for &line_i in &atom_line_i {
            let line = &self.lines[line_i];
            if line.len() < 80 {
                return Err(PDBError::InvalidFile("Line is too short".to_string()));
            }

            // Parse coordinates
            let x = parse_float_from_string(line, 30, 38)?;
            let y = parse_float_from_string(line, 38, 46)?;
            let z = parse_float_from_string(line, 46, 54)?;
            coords.push([x, y, z]);

            // Parse chain ID
            chain_ids.push(line[21..22].trim().to_string());

            // Parse residue ID
            res_ids.push(parse_number::<i32>(&line[22..26])?);

            // Parse residue name
            res_names.push(line[17..20].trim().to_string());

            // Parse hetero flag
            is_hetero.push(!line.starts_with("ATOM"));

            // Parse atom name
            atom_names.push(line[12..16].trim().to_string());

            // Parse element
            let element_str = line[76..78].trim();
            let element = if element_str.is_empty() {
                let atom_name = line[12..16].trim();
                if !atom_name.is_empty() {
                    let first_char = atom_name.chars().next().unwrap();
                    if first_char.is_alphabetic() && !first_char.is_numeric() {
                        Element::from_symbol(first_char.to_string()).unwrap()
                    } else {
                        Element::H // todo: fix
                    }
                } else {
                    Element::H // todo: fix
                }
            } else {
                Element::from_symbol(element_str).unwrap()
            };
            elements.push(element);

            // Parse atom ID
            atom_ids.push(parse_number::<i32>(&line[6..11])?);
        }
        // Parse bonds if available
        let bonds = self.parse_bonds(&atom_ids)?;
        let bonds_option = if bonds.is_empty() { None } else { Some(bonds) };
        let atom_collection = AtomCollection::new(
            size,
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            atom_names,
            chain_ids,
            bonds_option,
        );
        Ok(atom_collection)
    }

    /// Parse bonds from CONECT records
    fn parse_bonds(&self, atom_ids: &[i32]) -> Result<Vec<Bond>, PDBError> {
        // Mapping from atom ids to indices in an AtomArray
        let mut atom_id_to_index: HashMap<i32, i32> = HashMap::new();
        for (i, &id) in atom_ids.iter().enumerate() {
            atom_id_to_index.insert(id, i as i32);
        }
        let mut bonds: Vec<Bond> = Vec::new();
        for line in self.lines.iter() {
            if line.starts_with("CONECT") && line.len() >= 16 {
                // Extract ID of center atom
                if let Ok(center_id) = parse_number::<i32>(&line[6..11])
                    && let Some(&center_index) = atom_id_to_index.get(&center_id)
                {
                    // Iterate over atom IDs bonded to center atom
                    for i in (11..31).step_by(5) {
                        if i + 5 > line.len() {
                            break;
                        }
                        if let Ok(bonded_id) = parse_number::<i32>(&line[i..i + 5]) {
                            if let Some(&bonded_index) = atom_id_to_index.get(&bonded_id) {
                                bonds.push(Bond::new(
                                    center_index,
                                    bonded_index,
                                    BondOrder::Single,
                                ));
                            }
                        } else {
                            // No more bonded atoms
                            break;
                        }
                    }
                }
            }
        }
        Ok(bonds)
    }

    /// Write the `CRYST1` record to this [`PDBFile`] based on the given unit cell parameters.
    #[allow(dead_code)]
    pub fn write_box(
        &mut self,
        len_a: f32,
        len_b: f32,
        len_c: f32,
        alpha: f32,
        beta: f32,
        gamma: f32,
    ) {
        self.lines.push(format!(
            "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1          ",
            len_a, len_b, len_c, alpha, beta, gamma
        ));
    }

    /// Write data from an AtomCollection to this PDBFile
    #[allow(dead_code)]
    pub fn write_atom_collection(
        &mut self,
        atom_collection: &AtomCollection,
        as_multiple_models: bool,
    ) -> Result<(), PDBError> {
        let size = atom_collection.get_size();
        let coords = atom_collection.get_coords();

        // For single model mode, just write all atoms
        if !as_multiple_models {
            #[allow(clippy::needless_range_loop)] // i indexes several parallel per-atom columns
            for i in 0..size {
                let atom_line = format!(
                    "{:6}{:>5} {:4} {:>3} {:1}{:>4}{:1}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}  ",
                    if atom_collection.get_is_hetero(i) {
                        "HETATM"
                    } else {
                        "ATOM"
                    },
                    i + 1, // Atom ID
                    center_atom_name(
                        atom_collection.get_atom_name(i),
                        atom_collection.get_element(i)
                    ),
                    atom_collection.get_res_name(i),
                    atom_collection.get_chain_id(i),
                    atom_collection.get_res_id(i),
                    " ", // Insertion code
                    coords[i][0],
                    coords[i][1],
                    coords[i][2],
                    1.00, // Occupancy
                    0.00, // B-factor
                    atom_collection.get_element(i).to_string(),
                );
                self.lines.push(atom_line);
            }
        } else {
            // For multiple model mode, add MODEL/ENDMDL records
            // Note: This is a simplification that assumes a single model in the AtomCollection
            // In a real implementation, you'd need to handle multiple models in the collection
            self.lines.push(format!("MODEL {:>8}", 1));
            #[allow(clippy::needless_range_loop)] // i indexes several parallel per-atom columns
            for i in 0..size {
                let atom_line = format!(
                    "{:6}{:>5} {:4} {:>3} {:1}{:>4}{:1}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}  ",
                    if atom_collection.get_is_hetero(i) {
                        "HETATM"
                    } else {
                        "ATOM"
                    },
                    i + 1, // Atom ID
                    center_atom_name(
                        atom_collection.get_atom_name(i),
                        atom_collection.get_element(i)
                    ),
                    atom_collection.get_res_name(i),
                    atom_collection.get_chain_id(i),
                    atom_collection.get_res_id(i),
                    " ", // Insertion code
                    coords[i][0],
                    coords[i][1],
                    coords[i][2],
                    1.00, // Occupancy
                    0.00, // B-factor
                    atom_collection.get_element(i).to_string(),
                );
                self.lines.push(atom_line);
            }
            self.lines.push(String::from("ENDMDL"));
        }

        // Add bonds if available
        if let Some(bonds) = atom_collection.get_bonds() {
            self.write_bonds(bonds, size)?;
        }

        // Update the model and atom indices
        self.index_models_and_atoms();

        Ok(())
    }

    /// Write bonds to CONECT records
    fn write_bonds(&mut self, bonds: &[Bond], size: usize) -> Result<(), PDBError> {
        // Group bonds by center atom
        let mut bond_groups: HashMap<i32, Vec<i32>> = HashMap::new();
        for bond in bonds {
            let (center, bonded) = bond.get_atom_indices();
            bond_groups.entry(center).or_default().push(bonded);
            bond_groups.entry(bonded).or_default().push(center);
        }

        // Write CONECT records
        for center_idx in 0..size as i32 {
            if let Some(bonded_atoms) = bond_groups.get(&center_idx) {
                // We can only write up to 4 bonds per CONECT record
                for chunk in bonded_atoms.chunks(4) {
                    let mut line = format!("CONECT{:>5}", center_idx + 1); // 1-based atom IDs

                    for &bonded_idx in chunk {
                        line.push_str(&format!("{:>5}", bonded_idx + 1)); // 1-based atom IDs
                    }
                    // Pad to 80 characters
                    line = format!("{:<80}", line);
                    self.lines.push(line);
                }
            }
        }

        Ok(())
    }

    /// Index lines in the file that correspond to starts of new models and to
    /// `ATOM` or `HETATM` records.
    /// Must be called after the content of the file has been changed.
    fn index_models_and_atoms(&mut self) {
        self.atom_line_i = self
            .lines
            .iter()
            .enumerate()
            .filter(|(_i, line)| line.starts_with("ATOM") || line.starts_with("HETATM"))
            .map(|(i, _line)| i)
            .collect();
        self.model_start_i = self
            .lines
            .iter()
            .enumerate()
            .filter(|(_i, line)| line.starts_with("MODEL"))
            .map(|(i, _line)| i)
            .collect();
        // It could be an empty file or a file with a single model,
        // where the 'MODEL' line is missing
        if self.model_start_i.is_empty() && !self.atom_line_i.is_empty() {
            self.model_start_i = vec![0]
        }
    }
}

/// Center atom name for proper PDB formatting
/// If the element is a single character, the atom name needs to be centered differently
#[allow(dead_code)]
fn center_atom_name(atom_name: &str, element: &Element) -> String {
    if element.to_string().len() == 1 && atom_name.len() < 4 {
        format!(" {:<3}", atom_name)
    } else {
        format!("{:<4}", atom_name)
    }
}

/// Parse a string into a number.
/// Returns a `PDBError` if the parsing fails.
#[inline(always)]
fn parse_number<T: FromStr>(string: &str) -> Result<T, PDBError> {
    string.trim().parse().map_err(|_| {
        PDBError::InvalidFile(format!(
            "'{}' cannot be parsed into a number",
            string.trim()
        ))
    })
}

/// Parse a float from a specific region of a string
fn parse_float_from_string(line: &str, start: usize, stop: usize) -> Result<f32, PDBError> {
    if start >= line.len() || stop > line.len() || start >= stop {
        return Err(PDBError::InvalidFile("Invalid string range".to_string()));
    }

    line[start..stop].trim().parse().map_err(|_| {
        PDBError::InvalidFile(format!(
            "'{}' cannot be parsed into a float",
            line[start..stop].trim()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomCollection;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_pdb_file_read() {
        let (pdb_file, _temp) = TestFile::protein_02().create_temp().unwrap();
        let pdb = PDBFile::read(&pdb_file).unwrap();
        assert!(!pdb.lines.is_empty());
        assert!(!pdb.atom_line_i.is_empty());

        let ac: AtomCollection = pdb.parse_to_atom_collection().unwrap();
        assert_eq!(ac.get_size(), 1356);
        assert_eq!(
            ac.iter_chains()
                .map(|c| c.chain_id().to_string())
                .collect::<Vec<_>>(),
            ["A", "B", "C", "B", "C", "A", "B", "C"]
        );
    }

    // #[test]
    // fn test_pdb_from() {
    //     let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
    //     let (pdb_data, _) = pdbtbx::open(prot_file).unwrap();
    //     assert_eq!(pdb_data.atom_count(), 1413);

    //     // check Atom Collection Numbers
    //     let ac = AtomCollection::from(&pdb_data);
    //     assert_eq!(ac.get_coords().len(), 1413);
    //     assert_eq!(ac.get_bonds().unwrap().len(), 1095);

    //     // 338 Residues
    //     let res_ids: Vec<i32> = ac.get_resids().into_iter().cloned().unique().collect();
    //     let res_max = res_ids.iter().max().unwrap();
    //     assert_eq!(res_max, &338);

    //     // Check resnames
    //     let res_names: Vec<String> = ac
    //         .get_resnames()
    //         .into_iter()
    //         .cloned()
    //         .unique()
    //         .sorted()
    //         .collect();
    //     assert_eq!(
    //         res_names,
    //         [
    //             "ALA", "ARG", "ASN", "ASP", "GLN", "GLU", "GLY", "HEM", "HIS", "HOH", "ILE", "LEU",
    //             "LYS", "MET", "NBN", "PHE", "PRO", "SER", "SO4", "THR", "TRP", "TYR", "VAL"
    //         ]
    //     );

    //     // Take a peek at the unique elements
    //     let elements: Vec<Element> = ac
    //         .get_elements()
    //         .into_iter()
    //         .cloned()
    //         .unique()
    //         .sorted()
    //         .collect();
    //     assert_eq!(
    //         elements,
    //         [Element::C, Element::N, Element::O, Element::S, Element::Fe,]
    //     );
    // }
}
