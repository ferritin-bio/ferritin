//! Owned, boundary-ready snapshot of an [`AtomicHierarchy`].

use super::buffer::{Column, ColumnData};
use super::dictionary::StringDictionary;
use super::error::{BoundaryError, BoundaryErrorKind, to_u32};
use super::frame::FrameColumns;
use super::gpu::{BufferLayout, GpuBuffer};
use super::handles::{AtomHandle, ChainHandle, ResidueHandle};
use crate::model::{AtomicHierarchy, Model, ResidueGroup};
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

/// `atom.alt_loc` value for atoms without an alternate-location indicator.
pub const NO_ALT_LOC: u32 = 0;
/// `residue.ins_code` value for residues without an insertion code.
pub const NO_INS_CODE: u32 = 0;
/// `atom.formal_charge` value for atoms without a recorded formal charge.
pub const NO_FORMAL_CHARGE: i8 = i8::MIN;
/// `residue.group` value for polymer residues.
pub const RESIDUE_GROUP_POLYMER: u8 = 0;
/// `residue.group` value for non-polymer residues.
pub const RESIDUE_GROUP_NON_POLYMER: u8 = 1;

/// Boundary-ready, frame-independent topology columns.
///
/// Built once per `Arc<AtomicHierarchy>` and reused for every frame that shares
/// it (see [`Self::frame`]). Columns whose in-memory representation is already
/// a plain numeric array (sequence ids, segment offsets, bonds) are borrowed
/// from the hierarchy; everything else is encoded into owned buffers:
///
/// | Column | Type | Encoding |
/// |---|---|---|
/// | `atom.element` | `u8` | atomic number |
/// | `atom.alt_loc` | `u32` | Unicode scalar, [`NO_ALT_LOC`] if absent |
/// | `atom.formal_charge` | `i8` | charge, [`NO_FORMAL_CHARGE`] if absent |
/// | `residue.ins_code` | `u32` | Unicode scalar, [`NO_INS_CODE`] if absent |
/// | `residue.group` | `u8` | [`RESIDUE_GROUP_POLYMER`] / [`RESIDUE_GROUP_NON_POLYMER`] |
/// | `residue.label_seq_id` | `i32` | [`MISSING_SEQ_ID`](crate::model::MISSING_SEQ_ID) if absent |
/// | string fields | `u32` codes | [`StringDictionary`] |
///
/// # Ownership and invalidation
///
/// The snapshot holds a clone of the `Arc<AtomicHierarchy>` it was built from.
/// While it is alive, that hierarchy cannot be mutated in place: `Arc::get_mut`
/// fails and `Arc::make_mut` clones. So the snapshot never silently goes out of
/// date — a model whose hierarchy was replaced or copied-on-write simply stops
/// matching, which [`Self::is_current_for`] and [`Self::frame`] detect by
/// pointer identity.
#[derive(Clone, Debug)]
pub struct TopologyColumns {
    hierarchy: Arc<AtomicHierarchy>,
    n_atoms: u32,
    n_residues: u32,
    n_chains: u32,
    n_bonds: u32,
    atom_name: StringDictionary,
    auth_atom_name: StringDictionary,
    element: Vec<u8>,
    alt_loc: Vec<u32>,
    formal_charge: Vec<i8>,
    comp_id: StringDictionary,
    auth_comp_id: StringDictionary,
    ins_code: Vec<u32>,
    residue_group: Vec<u8>,
    label_asym_id: StringDictionary,
    auth_asym_id: StringDictionary,
    entity_id: StringDictionary,
}

fn encode_char(field: &str, row: usize, value: Option<char>) -> Result<u32, BoundaryError> {
    match value {
        None => Ok(0),
        Some('\0') => Err(BoundaryError::new(
            BoundaryErrorKind::InvalidModel,
            format!("{field}[{row}] is NUL, which is reserved for 'absent'"),
        )),
        Some(c) => Ok(c as u32),
    }
}

fn dictionary(values: &[String]) -> Result<StringDictionary, BoundaryError> {
    StringDictionary::encode(values.iter().map(String::as_str))
}

impl TopologyColumns {
    /// Validate `hierarchy` and encode its columns.
    pub fn encode(hierarchy: &Arc<AtomicHierarchy>) -> Result<Self, BoundaryError> {
        hierarchy.validate()?;
        let atoms = &hierarchy.atoms;
        let residues = &hierarchy.residues;
        let chains = &hierarchy.chains;

        let alt_loc = atoms
            .alt_loc
            .iter()
            .enumerate()
            .map(|(i, &c)| encode_char("atom.alt_loc", i, c))
            .collect::<Result<_, _>>()?;
        let formal_charge = atoms
            .formal_charge
            .iter()
            .enumerate()
            .map(|(i, &charge)| match charge {
                None => Ok(NO_FORMAL_CHARGE),
                Some(NO_FORMAL_CHARGE) => Err(BoundaryError::new(
                    BoundaryErrorKind::InvalidModel,
                    format!("atom.formal_charge[{i}] is {NO_FORMAL_CHARGE}, which is reserved for 'absent'"),
                )),
                Some(charge) => Ok(charge),
            })
            .collect::<Result<_, _>>()?;
        let ins_code = residues
            .ins_code
            .iter()
            .enumerate()
            .map(|(i, &c)| encode_char("residue.ins_code", i, c))
            .collect::<Result<_, _>>()?;
        let residue_group = residues
            .group
            .iter()
            .map(|group| match group {
                ResidueGroup::Polymer => RESIDUE_GROUP_POLYMER,
                ResidueGroup::NonPolymer => RESIDUE_GROUP_NON_POLYMER,
            })
            .collect();

        Ok(Self {
            hierarchy: Arc::clone(hierarchy),
            // `atom_bond_starts` holds n_atoms + 1 entries, so it bounds the atom count too.
            n_atoms: to_u32("atom count", atoms.len())?,
            n_residues: to_u32("residue count", residues.len())?,
            n_chains: to_u32("chain count", chains.len())?,
            n_bonds: to_u32("bond count", hierarchy.bonds.len())?,
            atom_name: dictionary(&atoms.atom_name)?,
            auth_atom_name: dictionary(&atoms.auth_atom_name)?,
            element: atoms
                .element
                .iter()
                .map(|e| e.atomic_number() as u8)
                .collect(),
            alt_loc,
            formal_charge,
            comp_id: dictionary(&residues.comp_id)?,
            auth_comp_id: dictionary(&residues.auth_comp_id)?,
            ins_code,
            residue_group,
            label_asym_id: dictionary(&chains.label_asym_id)?,
            auth_asym_id: dictionary(&chains.auth_asym_id)?,
            entity_id: dictionary(&chains.entity_id)?,
        })
    }

    /// Encode the topology of `model`.
    pub fn from_model(model: &Model) -> Result<Self, BoundaryError> {
        Self::encode(&model.hierarchy)
    }

    /// The hierarchy this snapshot was built from (and keeps alive).
    pub fn hierarchy(&self) -> &Arc<AtomicHierarchy> {
        &self.hierarchy
    }

    /// Whether `model` still uses the exact hierarchy this snapshot describes.
    pub fn is_current_for(&self, model: &Model) -> bool {
        Arc::ptr_eq(&self.hierarchy, &model.hierarchy)
    }

    /// Borrow the per-frame columns of `model`, checking it shares this topology.
    ///
    /// Fails with [`StaleTopology`](BoundaryErrorKind::StaleTopology) if it does not.
    pub fn frame<'m>(&self, model: &'m Model) -> Result<FrameColumns<'m>, BoundaryError> {
        if !self.is_current_for(model) {
            return Err(BoundaryError::new(
                BoundaryErrorKind::StaleTopology,
                "model does not share the hierarchy this topology snapshot was built from",
            ));
        }
        FrameColumns::new(model)
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> u32 {
        self.n_atoms
    }

    /// Number of residues.
    pub fn n_residues(&self) -> u32 {
        self.n_residues
    }

    /// Number of chains.
    pub fn n_chains(&self) -> u32 {
        self.n_chains
    }

    /// Number of bonds.
    pub fn n_bonds(&self) -> u32 {
        self.n_bonds
    }

    // --- atoms -------------------------------------------------------------

    /// `atom.name` (`label_atom_id`).
    pub fn atom_name(&self) -> &StringDictionary {
        &self.atom_name
    }

    /// `atom.auth_name` (`auth_atom_id`).
    pub fn auth_atom_name(&self) -> &StringDictionary {
        &self.auth_atom_name
    }

    /// `atom.element` as atomic numbers.
    pub fn element(&self) -> &[u8] {
        &self.element
    }

    /// `atom.alt_loc` as Unicode scalars ([`NO_ALT_LOC`] if absent).
    pub fn alt_loc(&self) -> &[u32] {
        &self.alt_loc
    }

    /// `atom.formal_charge` ([`NO_FORMAL_CHARGE`] if absent).
    pub fn formal_charge(&self) -> &[i8] {
        &self.formal_charge
    }

    // --- residues ----------------------------------------------------------

    /// `residue.comp_id` (`label_comp_id`).
    pub fn comp_id(&self) -> &StringDictionary {
        &self.comp_id
    }

    /// `residue.auth_comp_id`.
    pub fn auth_comp_id(&self) -> &StringDictionary {
        &self.auth_comp_id
    }

    /// `residue.label_seq_id` (borrowed from the hierarchy).
    pub fn label_seq_id(&self) -> &[i32] {
        &self.hierarchy.residues.label_seq_id
    }

    /// `residue.auth_seq_id` (borrowed from the hierarchy).
    pub fn auth_seq_id(&self) -> &[i32] {
        &self.hierarchy.residues.auth_seq_id
    }

    /// `residue.ins_code` as Unicode scalars ([`NO_INS_CODE`] if absent).
    pub fn ins_code(&self) -> &[u32] {
        &self.ins_code
    }

    /// `residue.group` codes.
    pub fn residue_group(&self) -> &[u8] {
        &self.residue_group
    }

    // --- chains ------------------------------------------------------------

    /// `chain.label_asym_id`.
    pub fn label_asym_id(&self) -> &StringDictionary {
        &self.label_asym_id
    }

    /// `chain.auth_asym_id`.
    pub fn auth_asym_id(&self) -> &StringDictionary {
        &self.auth_asym_id
    }

    /// `chain.entity_id`.
    pub fn entity_id(&self) -> &StringDictionary {
        &self.entity_id
    }

    // --- hierarchy segmentation and bonds (borrowed) -------------------------

    /// CSR offsets grouping atoms into residues (`n_residues + 1` entries).
    pub fn residue_atom_offsets(&self) -> &[u32] {
        self.hierarchy.atom_to_residue.offsets()
    }

    /// CSR offsets grouping residues into chains (`n_chains + 1` entries).
    pub fn chain_residue_offsets(&self) -> &[u32] {
        self.hierarchy.residue_to_chain.offsets()
    }

    /// First atom of each bond, sorted ascending.
    pub fn bond_atom_a(&self) -> &[u32] {
        &self.hierarchy.bonds.atom_a
    }

    /// Second atom of each bond.
    pub fn bond_atom_b(&self) -> &[u32] {
        &self.hierarchy.bonds.atom_b
    }

    /// Bond order of each bond.
    pub fn bond_order(&self) -> &[u8] {
        &self.hierarchy.bonds.order
    }

    /// CSR offsets grouping bonds by `atom_a` (`n_atoms + 1` entries).
    pub fn atom_bond_offsets(&self) -> &[u32] {
        &self.hierarchy.bonds.atom_bond_starts
    }

    /// Every numeric topology column, including dictionary codes, by stable name.
    ///
    /// Dictionary value tables are not listed; fetch them from the matching
    /// [`StringDictionary`] accessor.
    pub fn columns(&self) -> Vec<Column<'_>> {
        use ColumnData::{I8, I32, U8, U32};
        [
            ("atom.name", U32(self.atom_name.codes())),
            ("atom.auth_name", U32(self.auth_atom_name.codes())),
            ("atom.element", U8(&self.element)),
            ("atom.alt_loc", U32(&self.alt_loc)),
            ("atom.formal_charge", I8(&self.formal_charge)),
            ("residue.comp_id", U32(self.comp_id.codes())),
            ("residue.auth_comp_id", U32(self.auth_comp_id.codes())),
            ("residue.label_seq_id", I32(self.label_seq_id())),
            ("residue.auth_seq_id", I32(self.auth_seq_id())),
            ("residue.ins_code", U32(&self.ins_code)),
            ("residue.group", U8(&self.residue_group)),
            ("residue.atom_offsets", U32(self.residue_atom_offsets())),
            ("chain.label_asym_id", U32(self.label_asym_id.codes())),
            ("chain.auth_asym_id", U32(self.auth_asym_id.codes())),
            ("chain.entity_id", U32(self.entity_id.codes())),
            ("chain.residue_offsets", U32(self.chain_residue_offsets())),
            ("bond.atom_a", U32(self.bond_atom_a())),
            ("bond.atom_b", U32(self.bond_atom_b())),
            ("bond.order", U8(self.bond_order())),
            ("atom.bond_offsets", U32(self.atom_bond_offsets())),
        ]
        .into_iter()
        .map(|(name, data)| Column { name, data })
        .collect()
    }

    /// Bonds as an interleaved `[atom_a, atom_b]` `u32` index buffer, ready to
    /// use as a GPU line-list index buffer. Always an owned copy.
    pub fn bond_line_indices(&self) -> GpuBuffer<'static> {
        let bonds = &self.hierarchy.bonds;
        let mut bytes = Vec::with_capacity(bonds.len() * 8);
        for (&a, &b) in bonds.atom_a.iter().zip(&bonds.atom_b) {
            bytes.extend_from_slice(&a.to_le_bytes());
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        GpuBuffer {
            label: "bond.line_indices",
            layout: BufferLayout::new(super::ScalarType::U32, 2, 8, self.n_bonds),
            bytes: Cow::Owned(bytes),
        }
    }

    // --- checked handle navigation ------------------------------------------

    /// Residue containing `atom`.
    pub fn residue_of(&self, atom: AtomHandle) -> Result<ResidueHandle, BoundaryError> {
        if atom.0 >= self.n_atoms {
            return Err(BoundaryError::out_of_range("atom", atom.0, self.n_atoms));
        }
        Ok(ResidueHandle(
            self.hierarchy.residue_of_atom(atom.index()) as u32
        ))
    }

    /// Chain containing `residue`.
    pub fn chain_of(&self, residue: ResidueHandle) -> Result<ChainHandle, BoundaryError> {
        if residue.0 >= self.n_residues {
            return Err(BoundaryError::out_of_range(
                "residue",
                residue.0,
                self.n_residues,
            ));
        }
        Ok(ChainHandle(
            self.hierarchy.chain_of_residue(residue.index()) as u32,
        ))
    }

    /// Atom rows belonging to `residue`.
    pub fn atoms_of(&self, residue: ResidueHandle) -> Result<Range<u32>, BoundaryError> {
        let offsets = self.residue_atom_offsets();
        if residue.0 >= self.n_residues {
            return Err(BoundaryError::out_of_range(
                "residue",
                residue.0,
                self.n_residues,
            ));
        }
        Ok(offsets[residue.index()]..offsets[residue.index() + 1])
    }

    /// Residue rows belonging to `chain`.
    pub fn residues_of(&self, chain: ChainHandle) -> Result<Range<u32>, BoundaryError> {
        let offsets = self.chain_residue_offsets();
        if chain.0 >= self.n_chains {
            return Err(BoundaryError::out_of_range("chain", chain.0, self.n_chains));
        }
        Ok(offsets[chain.index()]..offsets[chain.index() + 1])
    }
}
