//! Low-level CIF file parsing and writing.
//!
//! Handles parsing of mmCIF/CIF files into atomic structures.

use crate::{AtomCollection, Bond};
use pdbtbx::Element;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::str::FromStr;

/// Custom error types for CIF parsing operations
#[derive(Debug)]
pub enum CIFError {
    InvalidFile(String),
    IOError(std::io::Error),
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
    name: String,
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

    /// Get value for a specific row and column
    fn get_value(&self, row: usize, column: &str) -> Option<&str> {
        if row >= self.data.len() {
            return None;
        }

        let col_idx = self.get_column_index(column)?;
        if col_idx >= self.data[row].len() {
            return None;
        }

        Some(&self.data[row][col_idx])
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
    fn parse_content(&mut self) -> Result<(), CIFError> {
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
                    name: line[5..].to_string(),
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
                        if let Some(block) = &mut current_data_block {
                            block.categories.insert(category.name.clone(), category);
                        }
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
                let parts: Vec<&str> = line.splitn(2, '.').collect();
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
                    let value = if i + 1 < lines.len() {
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

                        if j < line.len() {
                            values.push(line[start..j].to_string());
                            j += 1;
                        } else {
                            return Err(CIFError::InvalidFile("Unclosed quote".to_string()));
                        }
                    } else if c == ';' && j == 0 {
                        // Multi-line value
                        let mut value = String::new();
                        i += 1; // Move to next line

                        while i < lines.len() && !lines[i].trim_start().starts_with(';') {
                            value.push_str(lines[i]);
                            value.push('\n');
                            i += 1;
                        }

                        if i < lines.len() {
                            values.push(value);
                        } else {
                            return Err(CIFError::InvalidFile(
                                "Unclosed multi-line value".to_string(),
                            ));
                        }
                    } else {
                        // Regular value
                        let start = j;
                        while j < line.len() && !line.chars().nth(j).unwrap().is_whitespace() {
                            j += 1;
                        }

                        values.push(clean_cif_value(&line[start..j]));
                    }

                    j += 1;
                }

                // Add values to current row
                current_loop_row.extend(values);

                // If we have collected enough values for a row, add it to loop data
                if current_loop_row.len() >= loop_columns.len() {
                    loop_data.push(current_loop_row[0..loop_columns.len()].to_vec());
                    current_loop_row = current_loop_row[loop_columns.len()..].to_vec();
                }

                i += 1;
                continue;
            }

            // If we get here, it's an unrecognized line
            i += 1;
        }

        // Save final category if needed
        if let Some(mut category) = current_category.take() {
            if in_loop && !loop_data.is_empty() {
                category.columns = loop_columns;
                category.data = loop_data;
            }
            if let Some(block) = &mut current_data_block {
                block.categories.insert(category.name.clone(), category);
            }
        }

        // Save final data block if it exists
        if let Some(block) = current_data_block {
            self.data_blocks.push(block);
        }

        if self.data_blocks.is_empty() {
            return Err(CIFError::InvalidFile(
                "No data blocks found in CIF file".to_string(),
            ));
        }

        Ok(())
    }

    /// Get the number of data blocks in the file
    pub fn get_block_count(&self) -> usize {
        self.data_blocks.len()
    }

    /// Set the current data block
    pub fn set_current_block(&mut self, block_index: usize) -> Result<(), CIFError> {
        if block_index >= self.data_blocks.len() {
            return Err(CIFError::ValueError(format!(
                "Block index {} out of range (0-{})",
                block_index,
                self.data_blocks.len() - 1
            )));
        }
        self.current_block = block_index;
        Ok(())
    }

    /// Parse the crystallographic information to obtain the unit cell lengths
    /// and angles (in degrees).
    pub fn parse_box(&self) -> Result<Option<(f32, f32, f32, f32, f32, f32)>, CIFError> {
        if self.data_blocks.is_empty() {
            return Ok(None);
        }

        let block = &self.data_blocks[self.current_block];

        // Look for cell category
        if let Some(cell_category) = block.categories.get("cell") {
            let len_a = self.parse_category_value(cell_category, 0, "length_a")?;
            let len_b = self.parse_category_value(cell_category, 0, "length_b")?;
            let len_c = self.parse_category_value(cell_category, 0, "length_c")?;
            let alpha = self.parse_category_value(cell_category, 0, "angle_alpha")?;
            let beta = self.parse_category_value(cell_category, 0, "angle_beta")?;
            let gamma = self.parse_category_value(cell_category, 0, "angle_gamma")?;

            return Ok(Some((len_a, len_b, len_c, alpha, beta, gamma)));
        }

        Ok(None)
    }

    /// Helper to parse a numeric value from a category
    fn parse_category_value<T: FromStr>(
        &self,
        category: &CIFCategory,
        row: usize,
        column: &str,
    ) -> Result<T, CIFError> {
        let value_str = category.get_value(row, column).ok_or_else(|| {
            CIFError::InvalidFile(format!("Missing value for {}.{}", category.name, column))
        })?;

        value_str.parse().map_err(|_| {
            CIFError::InvalidFile(format!(
                "Failed to parse '{}' as a number for {}.{}",
                value_str, category.name, column
            ))
        })
    }

    /// Parse CIF file into an AtomCollection
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
        let res_name_col = atom_category
            .get_column_index("auth_comp_id")
            .or_else(|| atom_category.get_column_index("label_comp_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing residue name column".to_string()))?;
        let atom_name_col = atom_category
            .get_column_index("auth_atom_id")
            .or_else(|| atom_category.get_column_index("label_atom_id"))
            .ok_or_else(|| CIFError::InvalidFile("Missing atom name column".to_string()))?;
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

    /// Parse bonds from the CIF file
    fn parse_bonds(&self, atom_ids: &[i32]) -> Result<Vec<Bond>, CIFError> {
        let mut bonds = Vec::new();

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
    use ferritin_test_data::TestFile;

    #[test]
    fn test_cif_file_read() {
        let (cif_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let cif = CIFFile::read(&cif_file).unwrap();
        assert!(cif.data_blocks.len() > 0);

        // check conversion
        let ac: AtomCollection = cif.parse_to_atom_collection().unwrap();
        assert_eq!(ac.get_size(), 1413);
    }
}
