//! Low-level CIF file parsing and writing.
//!
//! Handles parsing of mmCIF/CIF files into atomic structures.

use crate::data::Segmentation;
use crate::info::elements::Element;
use crate::model::bonds::infer_bonds_from_residue_templates;
use crate::model::tables::{AtomsTable, ChainsTable, MISSING_SEQ_ID, ResidueGroup, ResiduesTable};
use crate::model::{
    Assembly, AssemblyUnit, AtomicConformation, AtomicHierarchy, CrystalSymmetry, IDENTITY_MAT4,
    Mat4, Model, SymmetryData, SymmetryOperator,
};
use crate::trajectory::ArrayTrajectory;
use crate::{AtomCollection, Bond};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::sync::Arc;

/// Custom error types for CIF parsing operations
#[derive(Debug)]
pub enum CIFError {
    InvalidFile(String),
    IOError(std::io::Error),
    #[allow(dead_code)]
    ValueError(String),
    OSError(String),
}

impl fmt::Display for CIFError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CIFError::InvalidFile(msg) => write!(f, "Invalid CIF file: {}", msg),
            CIFError::IOError(err) => write!(f, "IO error: {}", err),
            CIFError::ValueError(msg) => write!(f, "Value error: {}", msg),
            CIFError::OSError(msg) => write!(f, "OS error: {}", msg),
        }
    }
}

impl Error for CIFError {}

impl From<std::io::Error> for CIFError {
    fn from(err: std::io::Error) -> Self {
        CIFError::IOError(err)
    }
}

/// Represents a data block in a CIF file
#[derive(Debug)]
struct CIFDataBlock {
    _name: String,
    categories: HashMap<String, CIFCategory>,
}

/// Represents a category in a CIF file
#[derive(Debug)]
struct CIFCategory {
    name: String,
    columns: Vec<String>,
    data: Vec<Vec<String>>,
}

impl CIFCategory {
    /// Create a new CIF category
    fn new(name: String) -> Self {
        CIFCategory {
            name,
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Get column index by name
    fn get_column_index(&self, column_name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == column_name)
    }
}

/// This is a low-level abstraction of a CIF file.
/// The struct is able to parse coordinates, models, bonds etc. from CIF data.
pub struct CIFFile {
    /// The raw content of the CIF file
    raw_content: String,
    /// Parsed data blocks
    data_blocks: Vec<CIFDataBlock>,
    /// Current data block index
    current_block: usize,
}

impl CIFFile {
    /// Create a new [`CIFFile`] from raw content.
    pub fn new(content: String) -> Result<Self, CIFError> {
        let mut cif_file = CIFFile {
            raw_content: content,
            data_blocks: Vec::new(),
            current_block: 0,
        };
        cif_file.parse_content()?;
        Ok(cif_file)
    }

    /// Read a [`CIFFile`] from a file.
    /// The file is indicated by its file path.
    pub fn read<P: AsRef<std::path::Path>>(file_path: P) -> Result<Self, CIFError> {
        let path = file_path.as_ref();
        let content = fs::read_to_string(path)
            .map_err(|_| CIFError::OSError(format!("'{}' cannot be read", path.display())))?;
        Self::new(content)
    }

    /// Parse the raw content into structured data blocks
    pub(crate) fn parse_content(&mut self) -> Result<(), CIFError> {
        let mut current_data_block: Option<CIFDataBlock> = None;
        let mut current_category: Option<CIFCategory> = None;
        let mut in_loop = false;
        let mut loop_columns: Vec<String> = Vec::new();
        let mut loop_data: Vec<Vec<String>> = Vec::new();
        let mut current_loop_row: Vec<String> = Vec::new();

        let lines: Vec<&str> = self.raw_content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // A CIF `#` terminates the current loop/category.
            if line == "#" {
                if let Some(mut category) = current_category.take() {
                    if in_loop && !loop_data.is_empty() {
                        category.columns = loop_columns.clone();
                        category.data = loop_data.clone();
                    }
                    if let Some(block) = &mut current_data_block {
                        block.categories.insert(category.name.clone(), category);
                    }
                }
                in_loop = false;
                loop_columns.clear();
                loop_data.clear();
                current_loop_row.clear();
                i += 1;
                continue;
            }

            // Skip comments and empty lines
            if line.starts_with('#') || line.is_empty() {
                i += 1;
                continue;
            }

            // Handle data block
            if line.starts_with("data_") {
                // Save previous data block if it exists
                if let Some(block) = current_data_block.take() {
                    self.data_blocks.push(block);
                }
                // Create new data block
                current_data_block = Some(CIFDataBlock {
                    _name: line[5..].to_string(),
                    categories: HashMap::new(),
                });
                in_loop = false;
                i += 1;
                continue;
            }

            // Handle loop
            if line.starts_with("loop_") {
                // Finish previous category if needed
                if let Some(mut category) = current_category.take() {
                    if in_loop && !loop_data.is_empty() {
                        category.columns = loop_columns.clone();
                        category.data = loop_data.clone();
                    }
                    if let Some(block) = &mut current_data_block {
                        block.categories.insert(category.name.clone(), category);
                    }
                }
                in_loop = true;
                loop_columns = Vec::new();
                loop_data = Vec::new();
                current_loop_row = Vec::new();
                i += 1;
                continue;
            }

            // Handle loop column definitions (starting with underscore)
            if line.starts_with('_') {
                let mut tag_and_value = line.splitn(2, char::is_whitespace);
                let tag = tag_and_value.next().unwrap_or(line);
                let inline_value = tag_and_value
                    .next()
                    .map(str::trim)
                    .filter(|v| !v.is_empty());
                let parts: Vec<&str> = tag.splitn(2, '.').collect();
                if parts.len() < 2 {
                    return Err(CIFError::InvalidFile(format!(
                        "Invalid column definition: {}",
                        line
                    )));
                }
                let category_name = parts[0][1..].to_string(); // Remove the leading underscore
                let column_name = parts[1].trim().to_string();
                if in_loop {
                    // In loop context, collect column names
                    if current_category.is_none() {
                        current_category = Some(CIFCategory::new(category_name));
                    }
                    loop_columns.push(column_name);
                } else {
                    // Not in loop context, it's a single value
                    let value = if let Some(value) = inline_value {
                        clean_cif_value(value)
                    } else if i + 1 < lines.len() {
                        let next_line = lines[i + 1].trim();
                        if !next_line.starts_with('_')
                            && !next_line.starts_with("loop_")
                            && !next_line.starts_with("data_")
                        {
                            i += 1; // Skip the value line next iteration
                            clean_cif_value(next_line)
                        } else {
                            "".to_string() // No value
                        }
                    } else {
                        "".to_string() // No value
                    };

                    // Create or get category
                    if current_category.is_none()
                        || current_category.as_ref().unwrap().name != category_name
                    {
                        if let Some(category) = current_category.take() {
                            if let Some(block) = &mut current_data_block {
                                block.categories.insert(category.name.clone(), category);
                            }
                        }
                        current_category = Some(CIFCategory::new(category_name));
                    }

                    let category = current_category.as_mut().unwrap();
                    category.columns.push(column_name);
                    category.data.push(vec![value]);
                }

                i += 1;
                continue;
            }

            // Handle data rows in a loop
            if in_loop && !loop_columns.is_empty() {
                // Parse the line into values
                let mut values = Vec::new();
                let mut j = 0;

                while j < line.len() {
                    let c = line.chars().nth(j).unwrap();

                    if c.is_whitespace() {
                        j += 1;
                        continue;
                    }

                    if c == '\'' || c == '"' {
                        // Quoted value
                        let quote = c;
                        let start = j + 1;
                        j += 1;

                        while j < line.len() && line.chars().nth(j).unwrap() != quote {
                            j += 1;
                        }

                        let end = j;
                        values.push(line[start..end].to_string());
                        j += 1; // Skip closing quote
                    } else {
                        // Unquoted value
                        let start = j;
                        while j < line.len() && !line.chars().nth(j).unwrap().is_whitespace() {
                            j += 1;
                        }
                        let end = j;
                        values.push(clean_cif_value(&line[start..end]));
                    }
                }

                // Add values to current row
                current_loop_row.extend(values);

                // Check if we have a complete row
                if current_loop_row.len() >= loop_columns.len() {
                    loop_data.push(current_loop_row[..loop_columns.len()].to_vec());
                    current_loop_row = current_loop_row[loop_columns.len()..].to_vec();
                }
            }

            i += 1;
        }

        // Finish any pending category
        if let Some(mut category) = current_category.take() {
            if in_loop && !loop_data.is_empty() {
                category.columns = loop_columns;
                category.data = loop_data;
            }
            if let Some(block) = &mut current_data_block {
                block.categories.insert(category.name.clone(), category);
            }
        }

        // Add last data block
        if let Some(block) = current_data_block {
            self.data_blocks.push(block);
        }

        Ok(())
    }

    /// Parse CIF file into an AtomCollection (legacy API - returns flattened structure)
    pub fn parse_to_atom_collection(&self) -> Result<AtomCollection, CIFError> {
        if self.data_blocks.is_empty() {
            return Err(CIFError::InvalidFile("No data blocks found".to_string()));
        }

        let block = &self.data_blocks[self.current_block];

        // Find the atom_site category which contains atomic coordinates
        let atom_category = block
            .categories
            .get("atom_site")
            .ok_or_else(|| CIFError::InvalidFile("No atom_site category found".to_string()))?;

        let size = atom_category.data.len();
        let mut coords = Vec::with_capacity(size);
        let mut res_ids = Vec::with_capacity(size);
        let mut res_names = Vec::with_capacity(size);
        let mut is_hetero = Vec::with_capacity(size);
        let mut elements = Vec::with_capacity(size);
        let mut atom_names = Vec::with_capacity(size);
        let mut chain_ids = Vec::with_capacity(size);
        let mut atom_ids = Vec::with_capacity(size);

        // Get column indices
        let id_col = atom_category
            .get_column_index("id")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.id column".to_string()))?;
        let x_col = atom_category
            .get_column_index("Cartn_x")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_x column".to_string()))?;
        let y_col = atom_category
            .get_column_index("Cartn_y")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_y column".to_string()))?;
        let z_col = atom_category
            .get_column_index("Cartn_z")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_z column".to_string()))?;
        let chain_col = atom_category
            .get_column_index("auth_asym_id")
            .or_else(|| atom_category.get_column_index("label_asym_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing chain ID column".to_string()))?;
        let res_id_col = atom_category
            .get_column_index("auth_seq_id")
            .or_else(|| atom_category.get_column_index("label_seq_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing residue ID column".to_string()))?;
        let label_comp_col = atom_category
            .get_column_index("label_comp_id")
            .ok_or_else(|| CIFError::InvalidFile("Missing residue name column".to_string()))?;
        let auth_comp_col = atom_category
            .get_column_index("auth_comp_id")
            .unwrap_or(label_comp_col);
        let label_atom_col = atom_category
            .get_column_index("label_atom_id")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom name column".to_string()))?;
        let auth_atom_col = atom_category
            .get_column_index("auth_atom_id")
            .unwrap_or(label_atom_col);
        let res_name_col = auth_comp_col;
        let atom_name_col = auth_atom_col;
        let element_col = atom_category
            .get_column_index("type_symbol")
            .ok_or_else(|| CIFError::InvalidFile("Missing element column".to_string()))?;
        let group_pdb_col = atom_category.get_column_index("group_PDB");

        // Parse data rows
        for row in &atom_category.data {
            // Skip rows with less data than expected
            if row.len() <= x_col || row.len() <= y_col || row.len() <= z_col {
                continue;
            }

            // Parse coordinates
            let x = row[x_col].parse::<f32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid x coordinate: {}", row[x_col]))
            })?;
            let y = row[y_col].parse::<f32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid y coordinate: {}", row[y_col]))
            })?;
            let z = row[z_col].parse::<f32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid z coordinate: {}", row[z_col]))
            })?;
            coords.push([x, y, z]);

            // Parse atom ID
            let atom_id = row[id_col]
                .parse::<i32>()
                .map_err(|_| CIFError::InvalidFile(format!("Invalid atom ID: {}", row[id_col])))?;
            atom_ids.push(atom_id);

            // Parse chain ID
            chain_ids.push(row[chain_col].clone());

            // Parse residue ID
            let res_id = row[res_id_col].parse::<i32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid residue ID: {}", row[res_id_col]))
            })?;
            res_ids.push(res_id);

            // Parse residue name
            res_names.push(row[res_name_col].clone());

            // Parse atom name
            atom_names.push(row[atom_name_col].clone());

            // Parse element
            let element_str = &row[element_col];
            let element = Element::from_symbol(element_str).unwrap_or_else(|| {
                // Try to infer element from atom name if element is missing or invalid
                let atom_name = &row[atom_name_col];
                if atom_name.len() >= 1 {
                    let first_char = atom_name.chars().next().unwrap();
                    if first_char.is_alphabetic() && !first_char.is_numeric() {
                        Element::from_symbol(&first_char.to_string()).unwrap_or(Element::H)
                    } else {
                        Element::H
                    }
                } else {
                    Element::H
                }
            });
            elements.push(element);

            // Determine if atom is hetero
            let is_hetero_atom = if let Some(group_col) = group_pdb_col {
                if row.len() > group_col {
                    row[group_col] == "HETATM"
                } else {
                    false
                }
            } else {
                // If no group_PDB column, assume it's not a hetero atom
                false
            };
            is_hetero.push(is_hetero_atom);
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

    /// Parse CIF file into a single Model (first model only).
    /// For multi-model files, use `parse_to_trajectory`.
    pub fn parse_to_model(&self) -> Result<Model, CIFError> {
        if self.data_blocks.is_empty() {
            return Err(CIFError::InvalidFile("No data blocks found".to_string()));
        }

        let block = &self.data_blocks[self.current_block];
        let atom_category = block
            .categories
            .get("atom_site")
            .ok_or_else(|| CIFError::InvalidFile("No atom_site category found".to_string()))?;

        // Get column indices
        let model_num_col = atom_category.get_column_index("pdbx_PDB_model_num");
        let x_col = atom_category
            .get_column_index("Cartn_x")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_x column".to_string()))?;
        let y_col = atom_category
            .get_column_index("Cartn_y")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_y column".to_string()))?;
        let z_col = atom_category
            .get_column_index("Cartn_z")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_z column".to_string()))?;
        let chain_col = atom_category
            .get_column_index("auth_asym_id")
            .or_else(|| atom_category.get_column_index("label_asym_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing chain ID column".to_string()))?;
        let res_id_col = atom_category
            .get_column_index("auth_seq_id")
            .or_else(|| atom_category.get_column_index("label_seq_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing residue ID column".to_string()))?;
        let label_comp_col = atom_category
            .get_column_index("label_comp_id")
            .ok_or_else(|| CIFError::InvalidFile("Missing residue name column".to_string()))?;
        let auth_comp_col = atom_category
            .get_column_index("auth_comp_id")
            .unwrap_or(label_comp_col);
        let label_atom_col = atom_category
            .get_column_index("label_atom_id")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom name column".to_string()))?;
        let auth_atom_col = atom_category
            .get_column_index("auth_atom_id")
            .unwrap_or(label_atom_col);
        let element_col = atom_category
            .get_column_index("type_symbol")
            .ok_or_else(|| CIFError::InvalidFile("Missing element column".to_string()))?;
        let group_pdb_col = atom_category.get_column_index("group_PDB");
        let label_seq_id_col = atom_category.get_column_index("label_seq_id");
        let ins_code_col = atom_category.get_column_index("pdbx_PDB_ins_code");
        let label_asym_col = atom_category.get_column_index("label_asym_id");
        let entity_id_col = atom_category.get_column_index("label_entity_id");
        let occupancy_col = atom_category.get_column_index("occupancy");
        let b_iso_col = atom_category.get_column_index("B_iso_or_equiv");
        let formal_charge_col = atom_category.get_column_index("pdbx_formal_charge");

        // Select the lowest model number actually present; model numbering is
        // not required to begin at one.
        let mut first_model_num = 1;
        if let Some(col) = model_num_col {
            first_model_num = atom_category
                .data
                .iter()
                .map(|row| {
                    row.get(col)
                        .ok_or_else(|| CIFError::InvalidFile("Missing model number".to_string()))?
                        .parse::<i32>()
                        .map_err(|_| {
                            CIFError::InvalidFile(format!("Invalid model number: {}", row[col]))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .min()
                .ok_or_else(|| CIFError::InvalidFile("No atoms found".to_string()))?;
        }

        // Filter to first model only.
        let first_model_rows: Vec<&Vec<String>> = atom_category
            .data
            .iter()
            .filter(|row| {
                if let Some(col) = model_num_col {
                    if col < row.len() {
                        row[col].parse::<i32>().expect("model numbers validated") == first_model_num
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .collect();

        let n_atoms = first_model_rows.len();
        if n_atoms == 0 {
            return Err(CIFError::InvalidFile(
                "No atoms found in first model".to_string(),
            ));
        }

        // Build atom data
        let mut atom_names_vec: Vec<String> = Vec::with_capacity(n_atoms);
        let mut auth_atom_names_vec: Vec<String> = Vec::with_capacity(n_atoms);
        let mut elements_vec: Vec<Element> = Vec::with_capacity(n_atoms);
        let mut alt_loc_vec: Vec<Option<char>> = Vec::with_capacity(n_atoms);
        let mut formal_charge_vec: Vec<Option<i8>> = Vec::with_capacity(n_atoms);
        let mut x_vec: Vec<f32> = Vec::with_capacity(n_atoms);
        let mut y_vec: Vec<f32> = Vec::with_capacity(n_atoms);
        let mut z_vec: Vec<f32> = Vec::with_capacity(n_atoms);
        let mut occupancy_vec: Vec<f32> = Vec::with_capacity(n_atoms);
        let mut b_iso_vec: Vec<f32> = Vec::with_capacity(n_atoms);

        // For residue grouping
        let mut residue_keys: Vec<(String, i32, Option<char>)> = Vec::with_capacity(n_atoms);
        let mut chain_keys: Vec<String> = Vec::with_capacity(n_atoms);

        // For residue/chain tables
        let mut comp_ids: Vec<String> = Vec::new();
        let mut auth_comp_ids: Vec<String> = Vec::new();
        let mut label_seq_ids: Vec<i32> = Vec::new();
        let mut auth_seq_ids: Vec<i32> = Vec::new();
        let mut ins_codes: Vec<Option<char>> = Vec::new();
        let mut groups: Vec<ResidueGroup> = Vec::new();
        let mut label_asym_ids: Vec<String> = Vec::new();
        let mut auth_asym_ids: Vec<String> = Vec::new();
        let mut entity_ids: Vec<String> = Vec::new();

        let mut last_residue_key: Option<(String, i32, Option<char>)> = None;
        let mut last_chain_key: Option<String> = None;

        for row in &first_model_rows {
            // Parse coordinates
            let x = row[x_col].parse::<f32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid x coordinate: {}", row[x_col]))
            })?;
            let y = row[y_col].parse::<f32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid y coordinate: {}", row[y_col]))
            })?;
            let z = row[z_col].parse::<f32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid z coordinate: {}", row[z_col]))
            })?;
            x_vec.push(x);
            y_vec.push(y);
            z_vec.push(z);

            // Atom name
            atom_names_vec.push(row[label_atom_col].clone());
            auth_atom_names_vec.push(row[auth_atom_col].clone());

            // Element
            let element_str = &row[element_col];
            let element = Element::from_symbol(element_str).unwrap_or_else(|| {
                let first_char = element_str.chars().next().unwrap_or('C');
                Element::from_symbol(&first_char.to_string()).unwrap_or(Element::H)
            });
            elements_vec.push(element);

            // Alt loc (from label_alt_id if available)
            let alt_loc_col = atom_category.get_column_index("label_alt_id");
            let alt_loc = if let Some(col) = alt_loc_col {
                if col < row.len() {
                    let val = &row[col];
                    if val == "." || val.is_empty() {
                        None
                    } else {
                        val.chars().next()
                    }
                } else {
                    None
                }
            } else {
                None
            };
            alt_loc_vec.push(alt_loc);

            // Formal charge
            let formal_charge = if let Some(col) = formal_charge_col {
                if col < row.len() {
                    let val = &row[col];
                    if val == "?" || val == "." || val.is_empty() {
                        None
                    } else {
                        val.parse::<i8>().ok()
                    }
                } else {
                    None
                }
            } else {
                None
            };
            formal_charge_vec.push(formal_charge);

            // Occupancy
            let occupancy = if let Some(col) = occupancy_col {
                if col < row.len() {
                    row[col].parse::<f32>().map_err(|_| {
                        CIFError::InvalidFile(format!("Invalid occupancy: {}", row[col]))
                    })?
                } else {
                    1.0
                }
            } else {
                1.0
            };
            occupancy_vec.push(occupancy);

            // B factor
            let b_iso = if let Some(col) = b_iso_col {
                if col < row.len() {
                    row[col].parse::<f32>().map_err(|_| {
                        CIFError::InvalidFile(format!("Invalid B_iso_or_equiv: {}", row[col]))
                    })?
                } else {
                    0.0
                }
            } else {
                0.0
            };
            b_iso_vec.push(b_iso);

            // Chain and residue keys for grouping
            let auth_chain = row[chain_col].clone();
            let hierarchy_chain = label_asym_col
                .and_then(|col| row.get(col))
                .cloned()
                .unwrap_or_else(|| auth_chain.clone());
            let auth_seq_id = row[res_id_col].parse::<i32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid residue ID: {}", row[res_id_col]))
            })?;
            let ins_code = if let Some(col) = ins_code_col {
                if col < row.len() {
                    let val = &row[col];
                    if val == "?" || val == "." || val.is_empty() {
                        None
                    } else {
                        val.chars().next()
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let residue_key = (hierarchy_chain.clone(), auth_seq_id, ins_code);
            residue_keys.push(residue_key.clone());

            // Check if this is a new residue
            if last_residue_key.as_ref() != Some(&residue_key) {
                comp_ids.push(row[label_comp_col].clone());
                auth_comp_ids.push(row[auth_comp_col].clone());
                let label_seq = if let Some(col) = label_seq_id_col {
                    if col < row.len() {
                        if row[col].is_empty() {
                            MISSING_SEQ_ID
                        } else {
                            row[col].parse::<i32>().map_err(|_| {
                                CIFError::InvalidFile(format!("Invalid label_seq_id: {}", row[col]))
                            })?
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                label_seq_ids.push(label_seq);
                auth_seq_ids.push(auth_seq_id);
                ins_codes.push(ins_code);
                // chain_keys is per-residue (feeds residue_to_chain segmentation),
                // unlike residue_keys/chain_col reads above which are per-atom.
                chain_keys.push(hierarchy_chain.clone());

                // Determine group
                let is_hetero = if let Some(group_col) = group_pdb_col {
                    if group_col < row.len() {
                        row[group_col] == "HETATM"
                    } else {
                        false
                    }
                } else {
                    false
                };
                groups.push(if is_hetero {
                    ResidueGroup::NonPolymer
                } else {
                    ResidueGroup::Polymer
                });

                last_residue_key = Some(residue_key);
            }

            // Check if this is a new chain
            if last_chain_key.as_ref() != Some(&hierarchy_chain) {
                let label_asym = if let Some(col) = label_asym_col {
                    if col < row.len() {
                        row[col].clone()
                    } else {
                        auth_chain.clone()
                    }
                } else {
                    auth_chain.clone()
                };
                label_asym_ids.push(label_asym);
                auth_asym_ids.push(auth_chain.clone());

                let entity = if let Some(col) = entity_id_col {
                    if col < row.len() {
                        row[col].clone()
                    } else {
                        "1".to_string()
                    }
                } else {
                    "1".to_string()
                };
                entity_ids.push(entity);

                last_chain_key = Some(hierarchy_chain);
            }
        }

        validate_canonical_residue_order(&chain_keys, &auth_seq_ids, &ins_codes)?;

        // Build tables
        let atoms = AtomsTable {
            atom_name: atom_names_vec,
            auth_atom_name: auth_atom_names_vec,
            element: elements_vec,
            alt_loc: alt_loc_vec,
            formal_charge: formal_charge_vec,
        };

        let residues = ResiduesTable {
            comp_id: comp_ids,
            auth_comp_id: auth_comp_ids,
            label_seq_id: label_seq_ids,
            auth_seq_id: auth_seq_ids,
            ins_code: ins_codes,
            group: groups,
        };

        let chains = ChainsTable {
            label_asym_id: label_asym_ids,
            auth_asym_id: auth_asym_ids,
            entity_id: entity_ids,
        };

        // Build segmentations
        let atom_to_residue = Segmentation::from_change_points(residue_keys.iter());
        let residue_to_chain = Segmentation::from_change_points(chain_keys.iter());

        // Bonds are inferred from canonical-residue templates since mmCIF rarely
        // carries explicit connectivity records.
        let bonds = infer_bonds_from_residue_templates(
            &atoms,
            &residues,
            &atom_to_residue,
            &residue_to_chain,
        );

        let symmetry = Arc::new(parse_symmetry_data(block)?);
        let hierarchy = Arc::new(AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue,
            residue_to_chain,
            bonds,
        });

        let conformation = AtomicConformation {
            x: x_vec,
            y: y_vec,
            z: z_vec,
            occupancy: occupancy_col.map(|_| occupancy_vec),
            b_iso: b_iso_col.map(|_| b_iso_vec),
            confidence: None,
        };

        Model::try_new_with_symmetry(hierarchy, conformation, symmetry)
            .map_err(|error| CIFError::InvalidFile(error.to_string()))
    }

    /// Parse CIF file into an ArrayTrajectory (handles multi-model files).
    /// All models share the same AtomicHierarchy via Arc.
    pub fn parse_to_trajectory(&self) -> Result<ArrayTrajectory, CIFError> {
        if self.data_blocks.is_empty() {
            return Err(CIFError::InvalidFile("No data blocks found".to_string()));
        }

        let block = &self.data_blocks[self.current_block];
        let atom_category = block
            .categories
            .get("atom_site")
            .ok_or_else(|| CIFError::InvalidFile("No atom_site category found".to_string()))?;

        // Get column indices
        let model_num_col = atom_category.get_column_index("pdbx_PDB_model_num");
        let x_col = atom_category
            .get_column_index("Cartn_x")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_x column".to_string()))?;
        let y_col = atom_category
            .get_column_index("Cartn_y")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_y column".to_string()))?;
        let z_col = atom_category
            .get_column_index("Cartn_z")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom_site.Cartn_z column".to_string()))?;
        let chain_col = atom_category
            .get_column_index("auth_asym_id")
            .or_else(|| atom_category.get_column_index("label_asym_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing chain ID column".to_string()))?;
        let res_id_col = atom_category
            .get_column_index("auth_seq_id")
            .or_else(|| atom_category.get_column_index("label_seq_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing residue ID column".to_string()))?;
        let label_comp_col = atom_category
            .get_column_index("label_comp_id")
            .ok_or_else(|| CIFError::InvalidFile("Missing residue name column".to_string()))?;
        let auth_comp_col = atom_category
            .get_column_index("auth_comp_id")
            .unwrap_or(label_comp_col);
        let label_atom_col = atom_category
            .get_column_index("label_atom_id")
            .ok_or_else(|| CIFError::InvalidFile("Missing atom name column".to_string()))?;
        let auth_atom_col = atom_category
            .get_column_index("auth_atom_id")
            .unwrap_or(label_atom_col);
        let element_col = atom_category
            .get_column_index("type_symbol")
            .ok_or_else(|| CIFError::InvalidFile("Missing element column".to_string()))?;
        let group_pdb_col = atom_category.get_column_index("group_PDB");
        let label_seq_id_col = atom_category.get_column_index("label_seq_id");
        let ins_code_col = atom_category.get_column_index("pdbx_PDB_ins_code");
        let label_asym_col = atom_category.get_column_index("label_asym_id");
        let entity_id_col = atom_category.get_column_index("label_entity_id");
        let occupancy_col = atom_category.get_column_index("occupancy");
        let b_iso_col = atom_category.get_column_index("B_iso_or_equiv");
        let formal_charge_col = atom_category.get_column_index("pdbx_formal_charge");

        // Group rows by model number
        let mut model_rows: HashMap<i32, Vec<&Vec<String>>> = HashMap::new();
        for row in &atom_category.data {
            let model_num = if let Some(col) = model_num_col {
                let value = row
                    .get(col)
                    .ok_or_else(|| CIFError::InvalidFile("Missing model number".to_string()))?;
                value
                    .parse::<i32>()
                    .map_err(|_| CIFError::InvalidFile(format!("Invalid model number: {value}")))?
            } else {
                1
            };
            model_rows.entry(model_num).or_default().push(row);
        }

        if model_rows.is_empty() {
            return Err(CIFError::InvalidFile("No atoms found".to_string()));
        }

        // Get sorted model numbers
        let mut model_nums: Vec<i32> = model_rows.keys().copied().collect();
        model_nums.sort();

        // Use first model to build hierarchy
        let first_model_num = model_nums[0];
        let first_model_rows = &model_rows[&first_model_num];
        let n_atoms = first_model_rows.len();

        // Validate all models have same atom count
        for &model_num in &model_nums {
            let count = model_rows[&model_num].len();
            if count != n_atoms {
                return Err(CIFError::InvalidFile(format!(
                    "Model {} has {} atoms, but model {} has {} atoms. All models must have same atom count.",
                    model_num, count, first_model_num, n_atoms
                )));
            }
        }

        // Equal frame lengths are insufficient: coordinates may otherwise be
        // attached to the wrong atoms. Verify the complete identifier tuple in
        // the same order as the representative model.
        let identity_columns = [
            Some(label_atom_col),
            Some(auth_atom_col),
            Some(label_comp_col),
            Some(auth_comp_col),
            Some(chain_col),
            label_asym_col,
            Some(res_id_col),
            label_seq_id_col,
            ins_code_col,
            atom_category.get_column_index("label_alt_id"),
        ];
        for &model_num in &model_nums[1..] {
            for (atom_index, (expected, actual)) in first_model_rows
                .iter()
                .zip(&model_rows[&model_num])
                .enumerate()
            {
                let matches = identity_columns
                    .iter()
                    .flatten()
                    .all(|&column| expected.get(column) == actual.get(column));
                if !matches {
                    return Err(CIFError::InvalidFile(format!(
                        "Model {model_num} atom {atom_index} does not match model {first_model_num} identity/order"
                    )));
                }
            }
        }

        // Build hierarchy from first model
        let mut atom_names_vec: Vec<String> = Vec::with_capacity(n_atoms);
        let mut auth_atom_names_vec: Vec<String> = Vec::with_capacity(n_atoms);
        let mut elements_vec: Vec<Element> = Vec::with_capacity(n_atoms);
        let mut alt_loc_vec: Vec<Option<char>> = Vec::with_capacity(n_atoms);
        let mut formal_charge_vec: Vec<Option<i8>> = Vec::with_capacity(n_atoms);

        let mut residue_keys: Vec<(String, i32, Option<char>)> = Vec::with_capacity(n_atoms);
        let mut chain_keys: Vec<String> = Vec::with_capacity(n_atoms);

        let mut comp_ids: Vec<String> = Vec::new();
        let mut auth_comp_ids: Vec<String> = Vec::new();
        let mut label_seq_ids: Vec<i32> = Vec::new();
        let mut auth_seq_ids: Vec<i32> = Vec::new();
        let mut ins_codes: Vec<Option<char>> = Vec::new();
        let mut groups: Vec<ResidueGroup> = Vec::new();
        let mut label_asym_ids: Vec<String> = Vec::new();
        let mut auth_asym_ids: Vec<String> = Vec::new();
        let mut entity_ids: Vec<String> = Vec::new();

        let mut last_residue_key: Option<(String, i32, Option<char>)> = None;
        let mut last_chain_key: Option<String> = None;

        for row in first_model_rows {
            atom_names_vec.push(row[label_atom_col].clone());
            auth_atom_names_vec.push(row[auth_atom_col].clone());
            let element_str = &row[element_col];
            let element = Element::from_symbol(element_str).unwrap_or_else(|| {
                let first_char = element_str.chars().next().unwrap_or('C');
                Element::from_symbol(&first_char.to_string()).unwrap_or(Element::H)
            });
            elements_vec.push(element);

            let alt_loc_col = atom_category.get_column_index("label_alt_id");
            let alt_loc = if let Some(col) = alt_loc_col {
                if col < row.len() {
                    let val = &row[col];
                    if val == "." || val.is_empty() {
                        None
                    } else {
                        val.chars().next()
                    }
                } else {
                    None
                }
            } else {
                None
            };
            alt_loc_vec.push(alt_loc);

            let formal_charge = if let Some(col) = formal_charge_col {
                if col < row.len() {
                    let val = &row[col];
                    if val == "?" || val == "." || val.is_empty() {
                        None
                    } else {
                        val.parse::<i8>().ok()
                    }
                } else {
                    None
                }
            } else {
                None
            };
            formal_charge_vec.push(formal_charge);

            let auth_chain = row[chain_col].clone();
            let hierarchy_chain = label_asym_col
                .and_then(|col| row.get(col))
                .cloned()
                .unwrap_or_else(|| auth_chain.clone());
            let auth_seq_id = row[res_id_col].parse::<i32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid residue ID: {}", row[res_id_col]))
            })?;
            let ins_code = if let Some(col) = ins_code_col {
                if col < row.len() {
                    let val = &row[col];
                    if val == "?" || val == "." || val.is_empty() {
                        None
                    } else {
                        val.chars().next()
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let residue_key = (hierarchy_chain.clone(), auth_seq_id, ins_code);
            residue_keys.push(residue_key.clone());

            if last_residue_key.as_ref() != Some(&residue_key) {
                comp_ids.push(row[label_comp_col].clone());
                auth_comp_ids.push(row[auth_comp_col].clone());
                let label_seq = if let Some(col) = label_seq_id_col {
                    if col < row.len() {
                        if row[col].is_empty() {
                            MISSING_SEQ_ID
                        } else {
                            row[col].parse::<i32>().map_err(|_| {
                                CIFError::InvalidFile(format!("Invalid label_seq_id: {}", row[col]))
                            })?
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                label_seq_ids.push(label_seq);
                auth_seq_ids.push(auth_seq_id);
                ins_codes.push(ins_code);
                // chain_keys is per-residue (feeds residue_to_chain segmentation),
                // unlike residue_keys/chain_col reads above which are per-atom.
                chain_keys.push(hierarchy_chain.clone());

                let is_hetero = if let Some(group_col) = group_pdb_col {
                    if group_col < row.len() {
                        row[group_col] == "HETATM"
                    } else {
                        false
                    }
                } else {
                    false
                };
                groups.push(if is_hetero {
                    ResidueGroup::NonPolymer
                } else {
                    ResidueGroup::Polymer
                });

                last_residue_key = Some(residue_key);
            }

            if last_chain_key.as_ref() != Some(&hierarchy_chain) {
                let label_asym = if let Some(col) = label_asym_col {
                    if col < row.len() {
                        row[col].clone()
                    } else {
                        auth_chain.clone()
                    }
                } else {
                    auth_chain.clone()
                };
                label_asym_ids.push(label_asym);
                auth_asym_ids.push(auth_chain.clone());

                let entity = if let Some(col) = entity_id_col {
                    if col < row.len() {
                        row[col].clone()
                    } else {
                        "1".to_string()
                    }
                } else {
                    "1".to_string()
                };
                entity_ids.push(entity);

                last_chain_key = Some(hierarchy_chain);
            }
        }

        validate_canonical_residue_order(&chain_keys, &auth_seq_ids, &ins_codes)?;

        let atoms = AtomsTable {
            atom_name: atom_names_vec,
            auth_atom_name: auth_atom_names_vec,
            element: elements_vec,
            alt_loc: alt_loc_vec,
            formal_charge: formal_charge_vec,
        };

        let residues = ResiduesTable {
            comp_id: comp_ids,
            auth_comp_id: auth_comp_ids,
            label_seq_id: label_seq_ids,
            auth_seq_id: auth_seq_ids,
            ins_code: ins_codes,
            group: groups,
        };

        let chains = ChainsTable {
            label_asym_id: label_asym_ids,
            auth_asym_id: auth_asym_ids,
            entity_id: entity_ids,
        };

        let atom_to_residue = Segmentation::from_change_points(residue_keys.iter());
        let residue_to_chain = Segmentation::from_change_points(chain_keys.iter());
        // Bonds are inferred from canonical-residue templates since mmCIF rarely
        // carries explicit connectivity records.
        let bonds = infer_bonds_from_residue_templates(
            &atoms,
            &residues,
            &atom_to_residue,
            &residue_to_chain,
        );

        let symmetry = Arc::new(parse_symmetry_data(block)?);
        let hierarchy = Arc::new(AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue,
            residue_to_chain,
            bonds,
        });

        // Build models for each frame
        let mut models: Vec<Model> = Vec::with_capacity(model_nums.len());
        for &model_num in &model_nums {
            let rows = &model_rows[&model_num];
            let mut x_vec: Vec<f32> = Vec::with_capacity(n_atoms);
            let mut y_vec: Vec<f32> = Vec::with_capacity(n_atoms);
            let mut z_vec: Vec<f32> = Vec::with_capacity(n_atoms);
            let mut occupancy_vec: Vec<f32> = Vec::with_capacity(n_atoms);
            let mut b_iso_vec: Vec<f32> = Vec::with_capacity(n_atoms);

            for row in rows {
                let x = row[x_col].parse::<f32>().map_err(|_| {
                    CIFError::InvalidFile(format!("Invalid x coordinate: {}", row[x_col]))
                })?;
                let y = row[y_col].parse::<f32>().map_err(|_| {
                    CIFError::InvalidFile(format!("Invalid y coordinate: {}", row[y_col]))
                })?;
                let z = row[z_col].parse::<f32>().map_err(|_| {
                    CIFError::InvalidFile(format!("Invalid z coordinate: {}", row[z_col]))
                })?;
                x_vec.push(x);
                y_vec.push(y);
                z_vec.push(z);

                let occupancy = if let Some(col) = occupancy_col {
                    if col < row.len() {
                        row[col].parse::<f32>().map_err(|_| {
                            CIFError::InvalidFile(format!("Invalid occupancy: {}", row[col]))
                        })?
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                occupancy_vec.push(occupancy);

                let b_iso = if let Some(col) = b_iso_col {
                    if col < row.len() {
                        row[col].parse::<f32>().map_err(|_| {
                            CIFError::InvalidFile(format!("Invalid B_iso_or_equiv: {}", row[col]))
                        })?
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                b_iso_vec.push(b_iso);
            }

            let conformation = AtomicConformation {
                x: x_vec,
                y: y_vec,
                z: z_vec,
                occupancy: occupancy_col.map(|_| occupancy_vec),
                b_iso: b_iso_col.map(|_| b_iso_vec),
                confidence: None,
            };

            models.push(
                Model::try_new_with_symmetry(
                    Arc::clone(&hierarchy),
                    conformation,
                    Arc::clone(&symmetry),
                )
                .map_err(|error| CIFError::InvalidFile(error.to_string()))?,
            );
        }

        ArrayTrajectory::new(models).map_err(|e| CIFError::InvalidFile(e.to_string()))
    }

    /// Parse bonds from the CIF file
    fn parse_bonds(&self, atom_ids: &[i32]) -> Result<Vec<Bond>, CIFError> {
        let bonds = Vec::new();

        if self.data_blocks.is_empty() {
            return Ok(bonds);
        }

        let block = &self.data_blocks[self.current_block];

        // Create mapping from atom IDs to indices
        let mut atom_id_to_index: HashMap<i32, i32> = HashMap::new();
        for (i, &id) in atom_ids.iter().enumerate() {
            atom_id_to_index.insert(id, i as i32);
        }

        // Look for chem_comp_bond category
        if let Some(bond_category) = block.categories.get("struct_conn") {
            let atom1_col = bond_category.get_column_index("ptnr1_label_atom_id");
            let atom2_col = bond_category.get_column_index("ptnr2_label_atom_id");
            let res1_col = bond_category.get_column_index("ptnr1_label_seq_id");
            let res2_col = bond_category.get_column_index("ptnr2_label_seq_id");
            let chain1_col = bond_category.get_column_index("ptnr1_label_asym_id");
            let chain2_col = bond_category.get_column_index("ptnr2_label_asym_id");

            // We need enough columns to identify the atoms in a bond
            if atom1_col.is_none()
                || atom2_col.is_none()
                || res1_col.is_none()
                || res2_col.is_none()
                || chain1_col.is_none()
                || chain2_col.is_none()
            {
                return Ok(bonds); // Not enough info to parse bonds
            }

            // Find matching atoms and create bonds
            // This is complex and would require matching atoms by name/residue/chain
            // For now, return empty bonds as most applications will calculate bonds themselves
        }

        // Look for chem_comp_bond category
        // This would contain predefined bonds for each residue type
        // Again, complex to implement fully

        Ok(bonds)
    }
}

fn validate_canonical_residue_order(
    chain_ids: &[String],
    auth_seq_ids: &[i32],
    ins_codes: &[Option<char>],
) -> Result<(), CIFError> {
    let mut completed_chains = HashSet::new();
    for index in 1..chain_ids.len() {
        if chain_ids[index] == chain_ids[index - 1] {
            let previous = (auth_seq_ids[index - 1], ins_codes[index - 1]);
            let current = (auth_seq_ids[index], ins_codes[index]);
            if current < previous {
                return Err(CIFError::InvalidFile(format!(
                    "Residues in chain {} are not in canonical auth_seq_id/insertion-code order",
                    chain_ids[index]
                )));
            }
        } else {
            completed_chains.insert(chain_ids[index - 1].as_str());
            if completed_chains.contains(chain_ids[index].as_str()) {
                return Err(CIFError::InvalidFile(format!(
                    "Chain {} occurs in multiple non-contiguous blocks",
                    chain_ids[index]
                )));
            }
        }
    }
    Ok(())
}

fn required_column(category: &CIFCategory, name: &str) -> Result<usize, CIFError> {
    category
        .get_column_index(name)
        .ok_or_else(|| CIFError::InvalidFile(format!("Missing {}.{name} column", category.name)))
}

fn is_scalar_category(category: &CIFCategory) -> bool {
    category.data.len() == category.columns.len()
        && category.data.iter().all(|values| values.len() == 1)
}

fn category_row_count(category: &CIFCategory) -> usize {
    if is_scalar_category(category) {
        usize::from(!category.data.is_empty())
    } else {
        category.data.len()
    }
}

fn category_value(category: &CIFCategory, row: usize, column: usize) -> Option<&str> {
    if is_scalar_category(category) {
        if row == 0 {
            category.data.get(column)?.first().map(String::as_str)
        } else {
            None
        }
    } else {
        category.data.get(row)?.get(column).map(String::as_str)
    }
}

fn optional_scalar(block: &CIFDataBlock, category_name: &str, column: &str) -> Option<String> {
    let category = block.categories.get(category_name)?;
    let index = category.get_column_index(column)?;
    category_value(category, 0, index).map(str::to_string)
}

fn parse_operator_matrix(category: &CIFCategory, row: usize) -> Result<Mat4, CIFError> {
    let mut matrix = IDENTITY_MAT4;
    for row_index in 0..3 {
        for column_index in 0..3 {
            let name = format!("matrix[{}][{}]", row_index + 1, column_index + 1);
            let index = required_column(category, &name)?;
            let value = category_value(category, row, index)
                .ok_or_else(|| CIFError::InvalidFile(format!("Missing {name} value")))?;
            matrix[row_index][column_index] = value
                .parse::<f32>()
                .map_err(|_| CIFError::InvalidFile(format!("Invalid {name}: {value}")))?;
        }
        let name = format!("vector[{}]", row_index + 1);
        let index = required_column(category, &name)?;
        let value = category_value(category, row, index)
            .ok_or_else(|| CIFError::InvalidFile(format!("Missing {name} value")))?;
        matrix[row_index][3] = value
            .parse::<f32>()
            .map_err(|_| CIFError::InvalidFile(format!("Invalid {name}: {value}")))?;
    }
    Ok(matrix)
}

fn parse_fraction(value: &str) -> Result<f32, CIFError> {
    if let Some((numerator, denominator)) = value.split_once('/') {
        let numerator = numerator
            .parse::<f32>()
            .map_err(|_| CIFError::InvalidFile(format!("Invalid symmetry fraction: {value}")))?;
        let denominator = denominator
            .parse::<f32>()
            .map_err(|_| CIFError::InvalidFile(format!("Invalid symmetry fraction: {value}")))?;
        if denominator == 0.0 {
            return Err(CIFError::InvalidFile(format!(
                "Zero symmetry denominator: {value}"
            )));
        }
        Ok(numerator / denominator)
    } else {
        value
            .parse::<f32>()
            .map_err(|_| CIFError::InvalidFile(format!("Invalid symmetry value: {value}")))
    }
}

fn parse_symmetry_axis(expression: &str) -> Result<[f32; 4], CIFError> {
    let expression = expression.replace(' ', "").to_ascii_lowercase();
    let mut terms = Vec::new();
    let mut start = 0;
    for (index, character) in expression.char_indices().skip(1) {
        if character == '+' || character == '-' {
            terms.push(&expression[start..index]);
            start = index;
        }
    }
    terms.push(&expression[start..]);

    let mut result = [0.0; 4];
    for term in terms {
        if term.is_empty() {
            continue;
        }
        let (sign, unsigned) = match term.as_bytes()[0] {
            b'-' => (-1.0, &term[1..]),
            b'+' => (1.0, &term[1..]),
            _ => (1.0, term),
        };
        if let Some(axis) = unsigned.chars().last().filter(|axis| "xyz".contains(*axis)) {
            let coefficient = &unsigned[..unsigned.len() - 1];
            let coefficient = if coefficient.is_empty() {
                1.0
            } else {
                parse_fraction(coefficient)?
            };
            let index = match axis {
                'x' => 0,
                'y' => 1,
                'z' => 2,
                _ => unreachable!(),
            };
            result[index] += sign * coefficient;
        } else {
            result[3] += sign * parse_fraction(unsigned)?;
        }
    }
    Ok(result)
}

fn parse_xyz_operator(
    label: impl Into<String>,
    expression: &str,
) -> Result<SymmetryOperator, CIFError> {
    let axes = expression.split(',').collect::<Vec<_>>();
    if axes.len() != 3 {
        return Err(CIFError::InvalidFile(format!(
            "Symmetry operation must contain three axes: {expression}"
        )));
    }
    let mut matrix = IDENTITY_MAT4;
    for (index, axis) in axes.into_iter().enumerate() {
        matrix[index] = parse_symmetry_axis(axis)?;
    }
    Ok(SymmetryOperator::new(label, matrix))
}

fn fractional_to_cartesian_matrix(cell: [f32; 6]) -> Result<Mat4, CIFError> {
    let [a, b, c, alpha, beta, gamma] = cell;
    let (alpha, beta, gamma) = (alpha.to_radians(), beta.to_radians(), gamma.to_radians());
    let sin_gamma = gamma.sin();
    if a <= 0.0 || b <= 0.0 || c <= 0.0 || sin_gamma.abs() < 1e-7 {
        return Err(CIFError::InvalidFile(
            "Degenerate crystallographic unit cell".to_string(),
        ));
    }
    let c_x = c * beta.cos();
    let c_y = c * (alpha.cos() - beta.cos() * gamma.cos()) / sin_gamma;
    let c_z_squared = c * c - c_x * c_x - c_y * c_y;
    if c_z_squared <= 0.0 {
        return Err(CIFError::InvalidFile(
            "Degenerate crystallographic unit cell".to_string(),
        ));
    }
    Ok([
        [a, b * gamma.cos(), c_x, 0.0],
        [0.0, b * sin_gamma, c_y, 0.0],
        [0.0, 0.0, c_z_squared.sqrt(), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn invert_affine_basis(matrix: Mat4) -> Result<Mat4, CIFError> {
    let (a, b, c) = (matrix[0][0], matrix[0][1], matrix[0][2]);
    let (d, e, f) = (matrix[1][0], matrix[1][1], matrix[1][2]);
    let (g, h, i) = (matrix[2][0], matrix[2][1], matrix[2][2]);
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if determinant.abs() < 1e-10 {
        return Err(CIFError::InvalidFile(
            "Non-invertible crystallographic cell".to_string(),
        ));
    }
    let inverse = 1.0 / determinant;
    Ok([
        [
            (e * i - f * h) * inverse,
            (c * h - b * i) * inverse,
            (b * f - c * e) * inverse,
            0.0,
        ],
        [
            (f * g - d * i) * inverse,
            (a * i - c * g) * inverse,
            (c * d - a * f) * inverse,
            0.0,
        ],
        [
            (d * h - e * g) * inverse,
            (b * g - a * h) * inverse,
            (a * e - b * d) * inverse,
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn fractional_operator_to_cartesian(
    operator: SymmetryOperator,
    cell: [f32; 6],
) -> Result<SymmetryOperator, CIFError> {
    let fractional_to_cartesian = SymmetryOperator::new(
        "fractional_to_cartesian",
        fractional_to_cartesian_matrix(cell)?,
    );
    let cartesian_to_fractional = SymmetryOperator::new(
        "cartesian_to_fractional",
        invert_affine_basis(fractional_to_cartesian.matrix)?,
    );
    let label = operator.label.clone();
    Ok(fractional_to_cartesian
        .compose(&operator, "cell*symop")
        .compose(&cartesian_to_fractional, label))
}

fn expand_operator_group(group: &str) -> Result<Vec<String>, CIFError> {
    let mut result = Vec::new();
    for token in group
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if let Some((start, end)) = token.split_once('-') {
            let start = start
                .trim()
                .parse::<i32>()
                .map_err(|_| CIFError::InvalidFile(format!("Invalid operator range: {token}")))?;
            let end = end
                .trim()
                .parse::<i32>()
                .map_err(|_| CIFError::InvalidFile(format!("Invalid operator range: {token}")))?;
            if start > end {
                return Err(CIFError::InvalidFile(format!(
                    "Descending operator range: {token}"
                )));
            }
            result.extend((start..=end).map(|value| value.to_string()));
        } else {
            result.push(token.to_string());
        }
    }
    Ok(result)
}

fn operator_expression_groups(expression: &str) -> Result<Vec<Vec<String>>, CIFError> {
    let expression = expression.trim();
    if !expression.contains('(') {
        return Ok(vec![expand_operator_group(expression)?]);
    }
    let mut groups = Vec::new();
    let mut remainder = expression;
    while let Some(start) = remainder.find('(') {
        let after_start = &remainder[start + 1..];
        let end = after_start.find(')').ok_or_else(|| {
            CIFError::InvalidFile(format!("Unclosed operator expression: {expression}"))
        })?;
        groups.push(expand_operator_group(&after_start[..end])?);
        remainder = &after_start[end + 1..];
    }
    if groups.is_empty() || groups.iter().any(Vec::is_empty) {
        return Err(CIFError::InvalidFile(format!(
            "Empty operator expression: {expression}"
        )));
    }
    Ok(groups)
}

fn expand_operator_expression(
    expression: &str,
    operators: &HashMap<String, SymmetryOperator>,
) -> Result<Vec<SymmetryOperator>, CIFError> {
    let groups = operator_expression_groups(expression)?;
    let mut expanded = vec![SymmetryOperator::identity()];
    for group in groups {
        let mut next = Vec::new();
        for accumulated in &expanded {
            for id in &group {
                let operator = operators.get(id).ok_or_else(|| {
                    CIFError::InvalidFile(format!("Unknown symmetry operator {id}"))
                })?;
                let label = if accumulated.label == "identity" {
                    id.clone()
                } else {
                    format!("{}*{id}", accumulated.label)
                };
                next.push(accumulated.compose(operator, label));
            }
        }
        expanded = next;
    }
    Ok(expanded)
}

fn parse_symmetry_data(block: &CIFDataBlock) -> Result<SymmetryData, CIFError> {
    let mut operators = HashMap::new();
    if let Some(category) = block.categories.get("pdbx_struct_oper_list") {
        let id_column = required_column(category, "id")?;
        for row in 0..category_row_count(category) {
            let id = category_value(category, row, id_column)
                .ok_or_else(|| CIFError::InvalidFile("Missing operator id".to_string()))?
                .to_string();
            operators.insert(
                id.clone(),
                SymmetryOperator::new(id, parse_operator_matrix(category, row)?),
            );
        }
    }

    let details_by_id: HashMap<String, String> = block
        .categories
        .get("pdbx_struct_assembly")
        .map(|category| {
            let id = category.get_column_index("id")?;
            let details = category.get_column_index("details")?;
            Some(
                (0..category_row_count(category))
                    .filter_map(|row| {
                        Some((
                            category_value(category, row, id)?.to_string(),
                            category_value(category, row, details)?.to_string(),
                        ))
                    })
                    .collect(),
            )
        })
        .flatten()
        .unwrap_or_default();

    let mut assemblies_by_id: HashMap<String, Vec<AssemblyUnit>> = HashMap::new();
    if let Some(category) = block.categories.get("pdbx_struct_assembly_gen") {
        let assembly_column = required_column(category, "assembly_id")?;
        let expression_column = required_column(category, "oper_expression")?;
        let asym_column = required_column(category, "asym_id_list")?;
        for row in 0..category_row_count(category) {
            let asym_value = category_value(category, row, asym_column).ok_or_else(|| {
                CIFError::InvalidFile("Missing assembly asym_id_list".to_string())
            })?;
            let asym_ids = asym_value
                .split(',')
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            let expression = category_value(category, row, expression_column).ok_or_else(|| {
                CIFError::InvalidFile("Missing assembly oper_expression".to_string())
            })?;
            let assembly_id = category_value(category, row, assembly_column)
                .ok_or_else(|| CIFError::InvalidFile("Missing assembly_id".to_string()))?;
            for operator in expand_operator_expression(expression, &operators)? {
                assemblies_by_id
                    .entry(assembly_id.to_string())
                    .or_default()
                    .push(AssemblyUnit {
                        asym_ids: asym_ids.clone(),
                        operator,
                    });
            }
        }
    }
    let mut assemblies = assemblies_by_id
        .into_iter()
        .map(|(id, units)| Assembly {
            details: details_by_id.get(&id).cloned(),
            id,
            units,
        })
        .collect::<Vec<_>>();
    assemblies.sort_by(|a, b| a.id.cmp(&b.id));

    let cell_names = [
        "length_a",
        "length_b",
        "length_c",
        "angle_alpha",
        "angle_beta",
        "angle_gamma",
    ];
    let cell_values = cell_names
        .iter()
        .map(|name| optional_scalar(block, "cell", name))
        .collect::<Vec<_>>();
    let cell = if cell_values.iter().all(Option::is_some) {
        let mut values = [0.0; 6];
        for (index, value) in cell_values.into_iter().enumerate() {
            let value = value.unwrap();
            values[index] = value.parse::<f32>().map_err(|_| {
                CIFError::InvalidFile(format!("Invalid cell.{}: {value}", cell_names[index]))
            })?;
        }
        Some(values)
    } else {
        None
    };

    let space_group_name = optional_scalar(block, "symmetry", "space_group_name_H-M")
        .or_else(|| optional_scalar(block, "space_group", "name_H-M_alt"));

    let mut crystal_operators = Vec::new();
    for (category_name, operation_column) in [
        ("space_group_symop", "operation_xyz"),
        ("symmetry_equiv", "pos_as_xyz"),
    ] {
        if let Some(category) = block.categories.get(category_name) {
            let operation_index = required_column(category, operation_column)?;
            let id_index = category.get_column_index("id");
            for row_index in 0..category_row_count(category) {
                let label = id_index
                    .and_then(|index| category_value(category, row_index, index))
                    .map(str::to_string)
                    .unwrap_or_else(|| (row_index + 1).to_string());
                let expression =
                    category_value(category, row_index, operation_index).ok_or_else(|| {
                        CIFError::InvalidFile(
                            "Missing crystallographic symmetry operation".to_string(),
                        )
                    })?;
                let operator = parse_xyz_operator(label, expression)?;
                let cell = cell.ok_or_else(|| {
                    CIFError::InvalidFile(
                        "Crystallographic symmetry operators require complete cell parameters"
                            .to_string(),
                    )
                })?;
                crystal_operators.push(fractional_operator_to_cartesian(operator, cell)?);
            }
            break;
        }
    }

    Ok(SymmetryData {
        assemblies,
        crystal: CrystalSymmetry {
            space_group_name,
            cell,
            operators: crystal_operators,
        },
    })
}

/// Clean a CIF value (remove quotes, handle special values)
fn clean_cif_value(value: &str) -> String {
    let value = value.trim();

    // Handle quoted values
    if (value.starts_with('\'') && value.ends_with('\''))
        || (value.starts_with('"') && value.ends_with('"'))
    {
        return value[1..value.len() - 1].to_string();
    }

    // Handle special values
    if value == "." || value == "?" {
        return "".to_string();
    }

    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::Trajectory;
    use ferritin_test_data::TestFile;

    fn atom_site_cif(rows: &str) -> String {
        format!(
            "data_test\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.auth_atom_id\n_atom_site.label_comp_id\n_atom_site.auth_comp_id\n_atom_site.label_asym_id\n_atom_site.auth_asym_id\n_atom_site.label_entity_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num\n{rows}\n#\n"
        )
    }

    #[test]
    fn test_cif_file_read() {
        let (cif_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let cif = CIFFile::read(&cif_file).unwrap();
        assert!(cif.data_blocks.len() > 0);

        // check conversion
        let ac: AtomCollection = cif.parse_to_atom_collection().unwrap();
        assert_eq!(ac.get_size(), 1413);
    }

    #[test]
    fn test_cif_parse_to_model() {
        let (cif_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let cif = CIFFile::read(&cif_file).unwrap();
        let model = cif.parse_to_model().unwrap();
        assert_eq!(model.n_atoms(), 1413);
    }

    #[test]
    fn test_cif_parse_to_model_infers_bonds() {
        let (cif_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let cif = CIFFile::read(&cif_file).unwrap();
        let model = cif.parse_to_model().unwrap();
        assert!(
            !model.hierarchy.bonds.is_empty(),
            "parse_to_model should infer bonds from canonical-residue templates"
        );
        // Every atom should have at least one bond as either atom_a or atom_b.
        let n_atoms = model.n_atoms();
        let mut has_bond = vec![false; n_atoms];
        for i in 0..model.hierarchy.bonds.len() {
            has_bond[model.hierarchy.bonds.atom_a[i] as usize] = true;
            has_bond[model.hierarchy.bonds.atom_b[i] as usize] = true;
        }
        // Non-polymer atoms (waters, ions, ligands without a canonical-20 template)
        // are expected to stay unbonded, so this is a majority threshold rather than "all".
        let bonded_fraction = has_bond.iter().filter(|&&b| b).count() as f32 / n_atoms as f32;
        assert!(
            bonded_fraction > 0.75,
            "expected most polymer atoms to be bonded, got {:.2}",
            bonded_fraction
        );
    }

    #[test]
    fn test_cif_parse_to_model_distinct_chains() {
        // 4hhb has 4 protein chains (A/B/C/D), each followed later in the file by
        // its own hetero (heme) and water records under the *same* auth_asym_id
        // letter — so the hierarchy legitimately builds more than 4 contiguous
        // chain segments. What must NOT happen (see ferritin-ala.2) is every atom
        // collapsing onto the same chain letter: chain_keys was previously built
        // with one entry per atom rather than per residue, which fed
        // Segmentation::from_change_points atom-indexed offsets and made
        // residue_to_chain.segment_of(res_idx) return segment 0 for nearly all
        // valid residue indices, so every atom's derived chain_id string was "A".
        // AtomCollection::get_chain_id (via chain_of_residue) is exactly what the
        // ChainId color theme (ferritin-bevy colors.rs) reads per atom.
        let (cif_file, _temp) = TestFile::mvs_4hhb().create_temp().unwrap();
        let cif = CIFFile::read(&cif_file).unwrap();
        let model = cif.parse_to_model().unwrap();
        let ac = AtomCollection::from(&model);

        let assembly_id = model
            .symmetry
            .assemblies
            .first()
            .expect("4hhb should contain biological assembly metadata")
            .id
            .clone();
        let assembly_units = model.assembly_units(&assembly_id).unwrap();
        assert!(!assembly_units.is_empty());
        assert!(assembly_units
            .iter()
            .all(|unit| std::ptr::eq(unit.model(), &model)));

        let distinct_chain_ids: std::collections::HashSet<&str> =
            (0..ac.get_size()).map(|i| ac.get_chain_id(i)).collect();
        assert_eq!(
            distinct_chain_ids,
            std::collections::HashSet::from(["A", "B", "C", "D"]),
            "expected atoms to carry all 4 distinct chain letters, got {:?}",
            distinct_chain_ids
        );
    }

    #[test]
    fn test_cif_multimodel_trajectory() {
        let (cif_file, _temp) = TestFile::multimodel_nmr_1d3z().create_temp().unwrap();
        let cif = CIFFile::read(&cif_file).unwrap();
        let traj = cif.parse_to_trajectory().unwrap();

        // 1D3Z has 10 models, each with 1231 atoms
        assert_eq!(traj.frame_count(), 10);
        assert_eq!(traj.representative().n_atoms(), 1231);

        // All frames share the same hierarchy
        let h0 = &traj.frame(0).hierarchy;
        let h9 = &traj.frame(9).hierarchy;
        assert!(
            Arc::ptr_eq(h0, h9),
            "All frames must share the same Arc<AtomicHierarchy>"
        );
    }

    #[test]
    fn test_cif_multimodel_single_model_load() {
        let (cif_file, _temp) = TestFile::multimodel_nmr_1d3z().create_temp().unwrap();
        let cif = CIFFile::read(&cif_file).unwrap();
        let model = cif.parse_to_model().unwrap();

        // parse_to_model should return only the first model
        assert_eq!(model.n_atoms(), 1231);
    }

    #[test]
    fn test_cif_preserves_label_and_author_names_and_lowest_model() {
        let content = atom_site_cif(
            "ATOM 1 C CA CAA ALA ALX A AUTH 1 1 7 . 1.0 2.0 3.0 1.0 10.0 2\n\
             ATOM 2 C CA CAA ALA ALX A AUTH 1 1 7 . 4.0 5.0 6.0 1.0 10.0 3",
        );
        let cif = CIFFile::new(content).unwrap();
        let model = cif.parse_to_model().unwrap();
        assert_eq!(model.coord(0), [1.0, 2.0, 3.0]);
        assert_eq!(model.hierarchy.atoms.atom_name, ["CA"]);
        assert_eq!(model.hierarchy.atoms.auth_atom_name, ["CAA"]);
        assert_eq!(model.hierarchy.residues.comp_id, ["ALA"]);
        assert_eq!(model.hierarchy.residues.auth_comp_id, ["ALX"]);
    }

    #[test]
    fn test_cif_trajectory_rejects_mismatched_atom_identity() {
        let content = atom_site_cif(
            "ATOM 1 C CA CA ALA ALA A A 1 1 1 . 1.0 2.0 3.0 1.0 10.0 1\n\
             ATOM 2 C CB CB ALA ALA A A 1 1 1 . 4.0 5.0 6.0 1.0 10.0 2",
        );
        let error = CIFFile::new(content)
            .unwrap()
            .parse_to_trajectory()
            .err()
            .unwrap();
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn test_cif_rejects_malformed_required_residue_id() {
        let content = atom_site_cif("ATOM 1 C CA CA ALA ALA A A 1 1 nope . 1.0 2.0 3.0 1.0 10.0 1");
        let error = CIFFile::new(content)
            .unwrap()
            .parse_to_model()
            .err()
            .unwrap();
        assert!(error.to_string().contains("Invalid residue ID"));
    }

    #[test]
    fn test_cif_parses_and_expands_zero_copy_assembly() {
        let mut content =
            atom_site_cif("ATOM 1 C CA CA ALA ALA A A 1 1 1 . 1.0 2.0 3.0 1.0 10.0 1");
        content.push_str(
            "loop_\n_pdbx_struct_assembly.id\n_pdbx_struct_assembly.details\n1 'biological dimer'\n#\n\
             loop_\n_pdbx_struct_oper_list.id\n_pdbx_struct_oper_list.matrix[1][1]\n_pdbx_struct_oper_list.matrix[1][2]\n_pdbx_struct_oper_list.matrix[1][3]\n_pdbx_struct_oper_list.vector[1]\n_pdbx_struct_oper_list.matrix[2][1]\n_pdbx_struct_oper_list.matrix[2][2]\n_pdbx_struct_oper_list.matrix[2][3]\n_pdbx_struct_oper_list.vector[2]\n_pdbx_struct_oper_list.matrix[3][1]\n_pdbx_struct_oper_list.matrix[3][2]\n_pdbx_struct_oper_list.matrix[3][3]\n_pdbx_struct_oper_list.vector[3]\n\
             1 1 0 0 0 0 1 0 0 0 0 1 0\n\
             2 1 0 0 10 0 1 0 0 0 0 1 0\n#\n\
             loop_\n_pdbx_struct_assembly_gen.assembly_id\n_pdbx_struct_assembly_gen.oper_expression\n_pdbx_struct_assembly_gen.asym_id_list\n1 '(1,2)' A\n#\n",
        );
        let model = CIFFile::new(content).unwrap().parse_to_model().unwrap();
        let units = model.assembly_units("1").unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].atom_indices().collect::<Vec<_>>(), [0]);
        assert_eq!(units[0].coords().next().unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(units[1].coords().next().unwrap(), [11.0, 2.0, 3.0]);
        assert!(std::ptr::eq(units[0].model(), units[1].model()));
    }

    #[test]
    fn test_cif_parses_crystal_metadata_and_xyz_operators() {
        let mut content =
            atom_site_cif("ATOM 1 C CA CA ALA ALA A A 1 1 1 . 1.0 2.0 3.0 1.0 10.0 1");
        content.push_str(
            "_cell.length_a 10\n_cell.length_b 20\n_cell.length_c 30\n\
             _cell.angle_alpha 90\n_cell.angle_beta 90\n_cell.angle_gamma 90\n\
             _symmetry.space_group_name_H-M 'P 1'\n\
             loop_\n_space_group_symop.id\n_space_group_symop.operation_xyz\n1 x,y,z\n2 -x+1/2,y+1/2,-z\n#\n",
        );
        let model = CIFFile::new(content).unwrap().parse_to_model().unwrap();
        assert_eq!(
            model.symmetry.crystal.space_group_name.as_deref(),
            Some("P 1")
        );
        assert_eq!(
            model.symmetry.crystal.cell,
            Some([10.0, 20.0, 30.0, 90.0, 90.0, 90.0])
        );
        let units = model.crystal_units().unwrap();
        assert_eq!(units.len(), 2);
        let transformed = units[1].coords().next().unwrap();
        assert!((transformed[0] - 4.0).abs() < 1e-5);
        assert!((transformed[1] - 12.0).abs() < 1e-5);
        assert!((transformed[2] + 3.0).abs() < 1e-5);
    }
}
