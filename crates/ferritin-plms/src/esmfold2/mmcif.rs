//! mmCIF output generation for ESMFold2 predicted structures.
//!
//! Converts all-atom coordinate tensors and confidence scores to the mmCIF
//! format used by the Protein Data Bank. Output is compatible with
//! ferritin-core's mmCIF reader.
//!
//! ## Coordinate convention
//!
//! ESMFold2 outputs all-atom coordinates using the atom14 convention
//! (up to 14 heavy atoms per standard amino acid residue, in a fixed order).
//! This writer handles the backbone atoms (N, CA, C, O, CB) which are present
//! for all standard residues, plus any additional side-chain atoms at
//! non-zero positions.

use candle_core::Tensor;
use std::fmt::Write;

/// Standard 14-atom per residue heavy-atom names (atom14 convention).
/// Index i corresponds to the i-th atom slot.
pub const ATOM14_NAMES: [&str; 14] = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "NE", "CE", "CZ",
];

/// Element symbol for each atom14 slot (used for `type_symbol`).
pub const ATOM14_ELEMENTS: [&str; 14] = [
    "N", "C", "C", "O", "C", "C", "C", "C", "C", "C", "C", "N", "C", "C",
];

/// Three-letter residue name lookup from one-letter code.
pub fn one_to_three(aa: char) -> &'static str {
    match aa {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        _ => "UNK",
    }
}

/// Convert ESMFold2 output to mmCIF string (single model, single chain).
///
/// # Arguments
/// * `coords` — All-atom coordinates tensor. Accepts shapes:
///   - `(L, 14, 3)` (atom14, one residue per row)
///   - `(1, L, 14, 3)` (batched atom14; batch dim is squeezed)
///   - `(L, 3)` (Cα-only)
///   - `(1, L, 3)` (batched Cα-only; batch dim is squeezed)
/// * `sequence` — Protein sequence (one-letter codes), length `L`.
/// * `plddt` — Per-residue pLDDT in [0, 1], shape `(L,)` or `(1, L)`.
///   Written as B-factor × 100.
/// * `chain_id` — Chain identifier, e.g. `"A"`.
/// * `entry_id` — Data-block name, e.g. `"ESMFold2_pred"`.
///
/// # Returns
/// mmCIF text string compatible with ferritin-core's reader.
pub fn coords_to_mmcif(
    coords: &Tensor,
    sequence: &str,
    plddt: &Tensor,
    chain_id: &str,
    entry_id: &str,
) -> anyhow::Result<String> {
    // --- 1. Squeeze optional batch dimension ---
    let coords = if coords.rank() >= 3 && coords.dim(0)? == 1 {
        coords.squeeze(0)?
    } else {
        coords.clone()
    };

    let plddt = if plddt.rank() >= 2 && plddt.dim(0)? == 1 {
        plddt.squeeze(0)?
    } else {
        plddt.clone()
    };

    // --- 2. pLDDT → B-factor (×100) ---
    let plddt_f32 = plddt.to_dtype(candle_core::DType::F32)?;
    let plddt_vals = plddt_f32.to_vec1::<f32>()?;

    // --- 3. mmCIF header + loop_ block ---
    let mut out = String::new();

    writeln!(out, "data_{}", entry_id)?;
    writeln!(out, "#")?;
    writeln!(out, "loop_")?;
    writeln!(out, "_atom_site.group_PDB")?;
    writeln!(out, "_atom_site.id")?;
    writeln!(out, "_atom_site.type_symbol")?;
    writeln!(out, "_atom_site.label_atom_id")?;
    writeln!(out, "_atom_site.label_alt_id")?;
    writeln!(out, "_atom_site.label_comp_id")?;
    writeln!(out, "_atom_site.label_asym_id")?;
    writeln!(out, "_atom_site.label_entity_id")?;
    writeln!(out, "_atom_site.label_seq_id")?;
    writeln!(out, "_atom_site.pdbx_PDB_ins_code")?;
    writeln!(out, "_atom_site.Cartn_x")?;
    writeln!(out, "_atom_site.Cartn_y")?;
    writeln!(out, "_atom_site.Cartn_z")?;
    writeln!(out, "_atom_site.occupancy")?;
    writeln!(out, "_atom_site.B_iso_or_equiv")?;
    writeln!(out, "_atom_site.pdbx_formal_charge")?;
    writeln!(out, "_atom_site.auth_seq_id")?;
    writeln!(out, "_atom_site.auth_comp_id")?;
    writeln!(out, "_atom_site.auth_asym_id")?;
    writeln!(out, "_atom_site.auth_atom_id")?;
    writeln!(out, "_atom_site.pdbx_PDB_model_num")?;

    // --- 4. Atom rows ---
    let mut atom_id: u32 = 1;
    let l = sequence.len();

    let coords_f32 = coords.to_dtype(candle_core::DType::F32)?;

    match coords_f32.rank() {
        3 => {
            // Atom14 format: (L, 14, 3)
            let data = coords_f32.to_vec3::<f32>()?;
            for (i, aa) in sequence.chars().enumerate() {
                if i >= l {
                    break;
                }
                let res_name = one_to_three(aa);
                let seq_id = i + 1;
                let b_factor = plddt_vals.get(i).copied().unwrap_or(0.0) * 100.0;
                let natoms = data[i].len().min(ATOM14_NAMES.len());

                for j in 0..natoms {
                    let xyz = &data[i][j];
                    let (x, y, z) = (xyz[0], xyz[1], xyz[2]);

                    // Skip unoccupied atom14 slots (all near-zero)
                    if x.abs() + y.abs() + z.abs() < 0.001 {
                        continue;
                    }

                    let atom_name = ATOM14_NAMES[j];
                    let element = ATOM14_ELEMENTS[j];

                    writeln!(
                        out,
                        "ATOM {} {} {} . {} {} 1 {} ? {:.3} {:.3} {:.3} 1.00 {:.2} ? {} {} {} {} 1",
                        atom_id,
                        element,
                        atom_name,
                        res_name,
                        chain_id,
                        seq_id,
                        x,
                        y,
                        z,
                        b_factor,
                        seq_id,
                        res_name,
                        chain_id,
                        atom_name
                    )?;
                    atom_id += 1;
                }
            }
        }
        2 => {
            // Cα-only format: (L, 3)
            let data = coords_f32.to_vec2::<f32>()?;
            for (i, aa) in sequence.chars().enumerate() {
                if i >= l {
                    break;
                }
                let xyz = &data[i];
                let (x, y, z) = (xyz[0], xyz[1], xyz[2]);

                if x.abs() + y.abs() + z.abs() < 0.001 {
                    continue;
                }

                let res_name = one_to_three(aa);
                let seq_id = i + 1;
                let b_factor = plddt_vals.get(i).copied().unwrap_or(0.0) * 100.0;

                writeln!(
                    out,
                    "ATOM {} C CA . {} {} 1 {} ? {:.3} {:.3} {:.3} 1.00 {:.2} ? {} {} {} CA 1",
                    atom_id,
                    res_name,
                    chain_id,
                    seq_id,
                    x,
                    y,
                    z,
                    b_factor,
                    seq_id,
                    res_name,
                    chain_id
                )?;
                atom_id += 1;
            }
        }
        r => {
            return Err(anyhow::anyhow!(
                "Unexpected coords rank {r}. Expected 2 (Cα-only) or 3 (atom14).",
            ));
        }
    }

    writeln!(out, "#")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Build a small (2, 5, 3) coordinate tensor with recognizable values.
    fn make_coords(l: usize, natoms: usize) -> Tensor {
        let mut vals = Vec::with_capacity(l * natoms * 3);
        for i in 0..l {
            for j in 0..natoms {
                vals.push((i * 10 + j) as f32 + 1.0); // x
                vals.push((i * 10 + j) as f32 + 2.0); // y
                vals.push((i * 10 + j) as f32 + 3.0); // z
            }
        }
        Tensor::from_vec(vals, &[l, natoms, 3], &Device::Cpu).unwrap()
    }

    fn make_plddt(l: usize) -> Tensor {
        let vals: Vec<f32> = (0..l).map(|i| 0.7 + 0.01 * i as f32).collect();
        Tensor::from_vec(vals, &[l], &Device::Cpu).unwrap()
    }

    #[test]
    fn test_coords_to_mmcif_basic() {
        let sequence = "MA";
        let coords = make_coords(2, 5);
        let plddt = make_plddt(2);

        let result = coords_to_mmcif(&coords, sequence, &plddt, "A", "test_pred").unwrap();

        assert!(result.contains("data_test_pred"), "missing data_ header");
        assert!(result.contains("loop_"), "missing loop_");
        assert!(
            result.contains("_atom_site.Cartn_x"),
            "missing Cartn_x header"
        );
        assert!(result.contains("ATOM"), "missing ATOM records");
        // First residue is M → MET
        assert!(result.contains("MET"), "missing MET residue name");
    }

    #[test]
    fn test_coords_to_mmcif_ca_only() {
        let sequence = "GS";
        let vals: Vec<f32> = vec![
            1.0, 2.0, 3.0, // Gly CA
            4.0, 5.0, 6.0, // Ser CA
        ];
        let coords = Tensor::from_vec(vals, &[2, 3], &Device::Cpu).unwrap();
        let plddt = make_plddt(2);

        let result = coords_to_mmcif(&coords, sequence, &plddt, "B", "ca_only").unwrap();

        assert!(result.contains("data_ca_only"));
        assert!(result.contains("GLY"));
        assert!(result.contains("SER"));
        // Only CA atoms should appear
        let atom_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("ATOM")).collect();
        assert_eq!(atom_lines.len(), 2, "expected 2 CA atoms");
        for line in &atom_lines {
            assert!(line.contains(" CA "), "Cα-only: each atom should be CA");
        }
    }

    #[test]
    fn test_coords_to_mmcif_batched_squeeze() {
        // Shape (1, 2, 5, 3) — batch dim should be squeezed automatically
        let inner = make_coords(2, 5);
        let batched = inner.unsqueeze(0).unwrap(); // (1, 2, 5, 3)
        let plddt = make_plddt(2).unsqueeze(0).unwrap(); // (1, 2)

        let result = coords_to_mmcif(&batched, "MA", &plddt, "A", "batched").unwrap();

        assert!(result.contains("data_batched"));
        assert!(result.contains("ATOM"));
    }

    #[test]
    fn test_mmcif_output_scannable_for_atom_site() {
        // Verify the output can be line-scanned for _atom_site — mimicking what
        // ferritin-core's reader does before parsing.
        let sequence = "ACDE";
        let coords = make_coords(4, 5);
        let plddt = make_plddt(4);

        let result = coords_to_mmcif(&coords, sequence, &plddt, "A", "scan_test").unwrap();

        let has_loop = result.lines().any(|l| l == "loop_");
        let has_atom_site_col = result.lines().any(|l| l.starts_with("_atom_site."));
        let has_atom_record = result.lines().any(|l| l.starts_with("ATOM"));

        assert!(has_loop, "output must contain loop_");
        assert!(
            has_atom_site_col,
            "output must contain _atom_site.* headers"
        );
        assert!(has_atom_record, "output must contain ATOM records");

        // All ATOM lines should have 21 space-separated fields
        for line in result.lines().filter(|l| l.starts_with("ATOM")) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(
                fields.len(),
                21,
                "each ATOM line must have 21 fields, got {}: {:?}",
                fields.len(),
                line
            );
        }
    }

    #[test]
    fn test_one_to_three() {
        assert_eq!(one_to_three('A'), "ALA");
        assert_eq!(one_to_three('G'), "GLY");
        assert_eq!(one_to_three('W'), "TRP");
        assert_eq!(one_to_three('X'), "UNK");
    }

    #[test]
    fn test_skip_near_zero_atoms() {
        // Residue 0 has atoms at zero (should be skipped), residue 1 has real coords
        let vals: Vec<f32> = vec![
            // residue 0: 3 atoms all zero
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            // residue 1: 3 atoms with real coords
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ];
        let coords = Tensor::from_vec(vals, &[2, 3, 3], &Device::Cpu).unwrap();
        let plddt = make_plddt(2);

        let result = coords_to_mmcif(&coords, "AG", &plddt, "A", "zero_test").unwrap();

        let atom_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("ATOM")).collect();
        // Residue 0's 3 atoms are all zero → skipped. Residue 1 has 3 real atoms.
        assert_eq!(atom_lines.len(), 3, "only non-zero atoms should appear");
    }
}
