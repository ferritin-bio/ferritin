//! This module provides structures and implementations for handling PyMOL session data.
//!
//! It includes definitions for various PyMOL objects such as molecules, selectors,
//! and atom information. The module also provides functionality for converting
//! PyMOL data structures to PDB format using the `pdbtbx` crate.
//!
//! Key structures:
//! - `SessionName`: Represents a named session object
//! - `PyObjectMolecule`: Represents a molecule in a PyMOL session
//! - `SessionSelector`: Represents a selection in a PyMOL session
//! - `AtomInfo`: Contains detailed information about individual atoms
//!
//! This module is designed to work with serialized PyMOL session data and
//! provides methods for deserializing and manipulating this data.
//!
//! ## Links
//!
//! - [pymol-open-source](https://github.com/schrodinger/pymol-open-source). Pymol Source code.
//! - [Molecule Exporter](https://github.com/schrodinger/pymol-open-source/blob/master/layer3/MoleculeExporter.cpp#L1627)
//! - [PymolMoleculeExporter](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer4/Cmd.cpp#L3877)
//! - [PDB Exporter](https://github.com/schrodinger/pymol-open-source/blob/master/layer3/MoleculeExporter.cpp#L1627)
//! - [MoleculeExporterPDB](https://github.com/schrodinger/pymol-open-source/blob/master/layer3/MoleculeExporter.cpp#L439)
//! - [Gadget_01](https://github.com/schrodinger/pymol-open-source/blob/master/layer3/Executive.cpp#L5237)
//! - [Gadget_02](https://github.com/schrodinger/pymol-open-source/blob/master/layer2/ObjectGadget.cpp#L386)
//!
//! exporter->init(G);
//! exporter->setMulti(multi);
//! exporter->setRefObject(ref_object, ref_state);
//! exporter->execute(sele, state);
//!
//! CoordSetAtomToPDBStrVLA
//! Variables:
//! - m_iter: SeleCoordIterator m_iter;
//! Atoms:
//! how do we check for multiple objects?
//! m_iter.obj defines the object. By number? by name?
//! - iterate through the coordinates
//! - check for multi
//! update transformation matrices
//! updateMatrix(m_mat_full, true);
//! updateMatrix(m_mat_move, false);
//! beginCoordSet();
//! m_last_cs = m_iter.cs;
//! for bonds
//! if (!m_tmpids[m_iter.getAtm()]) {
//! m_id = m_retain_ids ? m_iter.getAtomInfo()->id : (m_id + 1);
//!  m_tmpids[m_iter.getAtm()] = m_id;
use crate::molviewspec::nodes::{ComponentExpression, ComponentSelector};
use crate::pymolparsing::colors::{Color, COLOR_SET};
use crate::pymolparsing::representation::RepBitmask;
use itertools::Itertools;
use pdbtbx::{self, Residue, PDB};
use serde::{Deserialize, Deserializer, Serialize};
use serde_pickle::{from_value, Value};
use serde_repr::{Deserialize_repr, Serialize_repr};

/// AtomInfo
///
/// This struct contains various properties of an atom, including its position,
/// chemical properties, and visualization settings.
///
/// ## Pymol Source
///
/// - [AtomInfo.h](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.h)
///     - Lots of goodies in here including constants defined for Atom type; Atom Geometry; VDW Radius; Chirality.
/// - [AtomInfo.cpp](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.cpp)
/// - [AtomInforAsPyList](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.cpp#L792)
/// - [VDW Radius of Elements](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.cpp#L1752)
///w
#[derive(Debug, Deserialize, Serialize)]
pub struct AtomInfo {
    /// Residue sequence number
    pub resv: i32,
    /// Chain identifier
    pub chain: String,
    /// Alternate location indicator
    pub alt: String,
    /// Residue identifier
    pub resi: String,
    /// Segment identifier
    pub segi: String,
    /// Residue name
    pub resn: String,
    /// Atom name
    pub name: String,
    /// Element symbol
    pub elem: String,
    /// Text type
    pub text_type: String,
    /// Label text
    pub label: String,
    /// Secondary structure type
    // todo: make enum
    pub ss_type: String,
    /// Flag indicating if the atom is hydrogen
    #[serde(deserialize_with = "int_to_bool")]
    pub is_hydrogen: bool,
    /// Custom type identifier
    pub custom_type: i32,
    /// Priority value
    pub priority: i32,
    /// B-factor (temperature factor)
    pub b: f64,
    /// Occupancy
    pub q: f64,
    /// Van der Waals radius
    pub vdw: f64,
    /// Partial charge
    pub partial_charge: f64,
    /// Formal charge
    pub formal_charge: i32,
    /// Flag indicating if the atom is a heteroatom
    #[serde(deserialize_with = "int_to_bool")]
    pub is_hetatm: bool,
    /// Visualization representation
    pub vis_rep: RepBitmask,
    /// Color index (color is an index into the color vec)
    pub color: i32,
    /// Atom ID
    pub id: i32,
    /// Cartoon representation type (0 = default which is auto (use ssType))
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.h#L292C33-L292C77
    pub cartoon: i32,
    /// Various flags
    pub flags: i64,
    /// Flag indicating if the atom is bonded
    #[serde(deserialize_with = "int_to_bool")]
    pub is_bonded: bool,
    /// Chemical flag (not sure what this is)
    pub chem_flag: i32,
    /// Geometry type (cAtomInfo*)
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.cpp#L44
    pub geom: i32,
    /// Valence (number of explicit and implicit neighbors, independent of bond order)
    // "valence" should be renamed to "degree" (or "total_degree"). It's the
    // number of explicit and implicit neighbors, independent of bond order.
    // Should be equivalent to RDKit::Atom::getTotalDegree() and
    // OBAtom::GetTotalDegree().
    pub valence: i32,
    /// Flag indicating if the atom is masked
    #[serde(deserialize_with = "int_to_bool")]
    pub is_masked: bool,
    /// Flag indicating if the atom is protected
    #[serde(deserialize_with = "int_to_bool")]
    pub is_protected: bool,
    /// Number of protons (atomic number)
    pub protons: i32,
    /// Unique identifier
    pub unique_id: i64,
    /// Stereochemistry indicator
    pub stereo: i8,
    /// Discrete state
    pub discrete_state: i32,
    /// Electronic radius
    pub elec_radius: f64,
    /// Rank
    pub rank: i32,
    /// Hydrogen bond donor flag
    #[serde(deserialize_with = "int_to_bool")]
    pub hb_donor: bool,
    /// Hydrogen bond acceptor flag
    #[serde(deserialize_with = "int_to_bool")]
    pub hb_acceptor: bool,
    /// Atomic color
    pub atomic_color: i32,
    /// Flag indicating if the atom has custom settings
    #[serde(deserialize_with = "int_to_bool")]
    pub has_setting: bool,
    /// Anisotropic temperature factor 1
    pub anisou_1: f32,
    /// Anisotropic temperature factor 2
    pub anisou_2: f32,
    /// Anisotropic temperature factor 3
    pub anisou_3: f32,
    /// Anisotropic temperature factor 4
    pub anisou_4: f32,
    /// Anisotropic temperature factor 5
    pub anisou_5: f32,
    /// Anisotropic temperature factor 6
    pub anisou_6: f32,
    /// Custom data string
    pub custom: String,
}
impl AtomInfo {
    /// is the atom a metal?
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.h#L319
    pub fn is_metal() {
        unimplemented!()
    }
    pub fn is_visible() {
        unimplemented!()
    }
    pub fn atom_info_brack_residue() {
        // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/AtomInfo.cpp#L1256C6-L1256C28
        //inefficient but reliable way to find where residue atoms are located in an object
        // for purpose of residue-based operations
        unimplemented!()
    }
    /// Conversion to PDBTBX
    pub fn to_pdbtbx_atom(&self) -> pdbtbx::Atom {
        let formal_charge = self.formal_charge as isize;
        let atom = pdbtbx::Atom::new(
            self.is_hetatm, // hetero
            0,              // serial_number
            &self.name,     // atom_name
            0.0,            // x Todo
            0.0,            // y Todo
            0.0,            // z Todo
            0.0,            // occupancy? Todo
            self.b,         // b-factor
            &self.elem,     // element
            formal_charge,  // charge: todo: is this the right charge?
        );
        atom.unwrap()
    }
}

/// Bond Structure
///
/// Represents a chemical bond between two atoms in a molecule.
///
#[derive(Debug, Deserialize, Serialize)]
pub struct Bond {
    /// Index of the first atom in the bond
    pub index_1: i32,
    /// Index of the second atom in the bond
    pub index_2: i32,
    /// Bond order (e.g., single, double, triple)
    // todo: make enum
    pub order: i32,
    /// Unique identifier for the bond
    pub id: i32,
    /// Stereochemistry information for the bond
    pub stereo: i32,
    /// Another unique identifier for the bond
    pub unique_id: i8,
    /// Flag indicating if the bond has custom settings
    pub has_setting: i32,
    // todo handle arity 7 or arity 8 with specific symmetry info
    // Symmetry operation of the second atom.
    // symop_2: Option<String>,
}

/// Coord Set: Class for storage of coordinates
///
/// Links:
/// - [pymol_coordset](https://github.com/schrodinger/pymol-open-source/blob/master/layer2/CoordSet.cpp#L363)
/// - [pymol_coordset_settings]( https://github.com/schrodinger/pymol-open-source/blob/master/layer1/Setting.cpp#L962)
/// - [CGO](https://github.com/schrodinger/pymol-open-source/blob/master/layer1/CGO.cpp#L220)
/// - [symettry_settings](https://github.com/schrodinger/pymol-open-source/blob/master/layer1/Symmetry.cpp#L30)
///
#[derive(Debug, Deserialize, Serialize)]
pub struct CoordSet {
    pub n_index: i32,         // 1519
    n_at_index: i32,          // 1519
    pub coord: Vec<f32>,      // len -== 4556 ( 1519 *3 )
    pub idx_to_atm: Vec<i32>, // 1 - 1518
    pub atm_to_idx: Option<Vec<i32>>,
    pub name: String,
    pub setting: Vec<Option<bool>>,          // punting on this
    pub lab_pos: Option<bool>,               // might be wrong
    field_9: Option<bool>,                   // probably wrong...
    pub sculpt_cgo: Option<(i32, Vec<f32>)>, //
    pub atom_state_settings: Option<Vec<Option<i32>>>, //
    pub symmetry: Option<Vec<(((i32, i32, i32), (i32, i32, i32)), String)>>,
}

impl CoordSet {
    /// coords are stored in a 1D vector of x,y,z,x,y,x,z,x,y,z
    ///returns as a Vec<[x,y,z]>
    pub fn get_coords_as_vec(&self) -> Vec<[f32; 3]> {
        self.coord
            .chunks(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect()
    }
}
/// Custom Value
///
/// needed for the settings triplet.
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum CustomValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}

// https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer1/PyMOLObject.h#L39
#[derive(Serialize_repr, Deserialize_repr, PartialEq, Debug, Clone)]
#[repr(i32)]
/// Pymol Object Types
pub enum ObjectType {
    CObjectMolecule = 1,
    CObjectMap = 2,
    CObjectMesh = 3,
    CObjectMeasurement = 4,
    CObjectCallback = 5,
    CObjectCGO = 6,
    CObjectSurface = 7,
    CObjectGadget = 8,
    CObjectCalculator = 9,
    CObjectSlice = 10,
    CObjectAlignment = 11,
    CObjectGroup = 12,
    CObjectVolume = 13,
    CObjectCurve = 14,
}

/// PyObject
///
/// General Object-Level settings object
///
/// - [PyMOLObject](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer1/PyMOLObject.h#L189)
///
#[derive(Debug, Deserialize, Serialize)]
pub struct PyObject {
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer1/PyMOLObject.h#L39
    pub object_type: ObjectType, // cObject_t
    pub name: String,
    pub color: i32, // represents a pointer into an array of colors.
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer1/Rep.h
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/modules/pymol/constants.py#L155-L177
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/modules/pymol/viewing.py#L51C1-L53C71
    pub vis_rep: RepBitmask, //
    pub extent_min: [f32; 3],
    pub extent_max: [f32; 3],
    pub extent_flag: i32,
    pub ttt_flag: i32,
    pub setting: Option<bool>, // this is a hack
    pub enabled: i32,
    pub render_context: i32,
    pub ttt: [f32; 16], //  float TTT[16]{}; /* translate, transform, translate matrix (to apply when rendering)
    pub n_frame: i32,
    pub view_elem: Option<bool>, // /* for animating objects via the TTT */
}
impl PyObject {
    /// get_color - note this currently works if there are
    /// no custom colors.
    pub fn get_color(&self) -> Color {
        COLOR_SET
            .get(self.color as usize)
            .expect("Index within bounds")
            .clone()
    }
}
/// PyObjectMolecule: Represents a molecule in PyMOL.
///
/// ## Link
///
/// - [pymol code](https://github.com/schrodinger/pymol-open-source/blob/master/layer2/ObjectMolecule2.cpp#L3524)
/// - [ObjectMolecule](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer2/ObjectMolecule.h#L58)
/// - [Bond](https://github.com/schrodinger/pymol-open-source/blob/master/layer2/ObjectMolecule2.cpp#L3037)
/// - [AtomInfo](https://github.com/schrodinger/pymol-open-source/blob/master/layer2/ObjectMolecule2.cpp#L3248)
/// - [AtomInfo2](https://github.com/schrodinger/pymol-open-source/blob/master/layer2/AtomInfo.cpp#L792)
///
#[derive(Debug, Deserialize, Serialize)]
pub struct PyObjectMolecule {
    /// The base PyObject information
    pub object: PyObject,
    /// Number of coordinate sets
    pub n_cset: i32,
    /// Number of bonds
    pub n_bond: i32,
    /// Number of atoms
    pub n_atom: i32,
    /// Vector of coordinate sets
    pub coord_set: Vec<CoordSet>,
    /// Optional template coordinate set
    pub cs_tmpl: Option<Vec<CoordSet>>,
    /// Vector of bonds
    pub bond: Vec<Bond>,
    /// Vector of atom information
    pub atom: Vec<AtomInfo>,
    /// Flag for discrete representation
    pub discrete_flag: i32,
    /// Number of discrete objects
    pub n_discrete: i32,
    /// Optional symmetry information (crystal space group and name)
    pub symmetry: Option<(([f32; 3], [f32; 3]), String)>,
    /// Current coordinate set index
    pub cur_cset: i32,
    /// Counter for bonds
    pub bond_counter: i32,
    /// Counter for atoms
    pub atom_counter: i32,
    /// Optional mapping of discrete atoms to indices
    pub discrete_atm_to_idx: Option<Vec<i32>>,
    /// Optional discrete coordinate set information
    pub dcs: Option<Vec<i32>>,
}
impl PyObjectMolecule {
    pub fn get_name(&self) -> String {
        self.object.name.to_string()
    }
    /// Create a PDBTBX::Atom from the pymol object datastructure
    pub fn get_atom(&self, atm_idx: i32) -> pdbtbx::Atom {
        // find atom ids and coordinates in the CoordSet
        // find the remaining atom info in the AtomInfo Vector
        let cset = &self.coord_set;
        // println!("{:?}", cset);
        let atom_coords = &cset[0].coord; // note there may be more than one coord set.... Todo.

        // println!("{:?}", atom_coords);
        // coords are stored in a 1D vector of x,y,z,x,y,x,z,x,y,z
        let base_coord = (3 * atm_idx) as usize;
        // println!("{}", atm_idx);
        // println!("{}", base_coord);
        let x_coord = atom_coords[base_coord];
        // println!("{}", x_coord);
        let y_coord = atom_coords[base_coord + 1];
        // println!("{}", y_coord);
        let z_coord = atom_coords[base_coord + 2];
        // println!("{}", z_coord);
        // println!("{}, {}, {}", x_coord, y_coord, z_coord);
        let atom_info = &self.atom.iter().find(|atm| atm.id == atm_idx + 1).unwrap(); // note that the atom in the atom vector seem to be 1-indexed.
        let formal_charge = atom_info.formal_charge as isize;
        let serial_number = atom_info.id as usize;

        let atom = pdbtbx::Atom::new(
            atom_info.is_hetatm,    // hetero
            serial_number,          // serial_number: Note: I am not sure this is correct just yet.
            atom_info.name.clone(), // atom_name
            x_coord.into(),         // x
            y_coord.into(),         // y
            z_coord.into(),         // z
            0.0,                    // occupancy? Todo
            atom_info.b,            // b-factor
            atom_info.elem.clone(), // element
            formal_charge,          // charge: todo: is this the right charge?
        );
        atom.unwrap()
    }

    /// Get unique chain names
    pub fn get_chains(&self) -> Vec<String> {
        self.atom
            .iter()
            .filter_map(|atm| Some(atm.chain.clone()))
            .unique() // from itertools
            .collect()
    }
    /// Get each residue by chain.
    pub fn get_residues_by_chain(&self, chain: String) -> Vec<i32> {
        self.atom
            .iter()
            .filter(|atm| atm.chain == chain)
            .map(|atm| atm.resv)
            .unique()
            .collect()
    }
    /// Unit Cell Symetry.
    pub fn get_unit_cell_symmetry(&self) -> (pdbtbx::UnitCell, pdbtbx::Symmetry) {
        let symmetry = &self.symmetry.clone().expect("Expected a symmetry group.");
        let (([a, b, c], [alpha, beta, gamma]), sym_group) = symmetry;
        println!(
            "{}, {}, {}, {}, {}, {}, {}",
            a, b, c, alpha, beta, gamma, sym_group
        );

        // let unitcell = pdbtbx::UnitCell::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let unitcell = pdbtbx::UnitCell::new(
            *a as f64,
            *b as f64,
            *c as f64,
            *alpha as f64,
            *beta as f64,
            *gamma as f64,
        );

        let pdbsym = pdbtbx::Symmetry::new(sym_group).expect("Invalid Symmetry group");

        (unitcell, pdbsym)
    }
    /// Get each residue by chain.
    pub fn create_residue(&self, chain: String, residue_number: i32) -> pdbtbx::Residue {
        let atoms: Vec<&AtomInfo> = self
            .atom
            .iter()
            .filter(|atm| atm.chain == chain && atm.resv == residue_number)
            .collect();

        let resv = residue_number as isize;
        let res_name = atoms[0].resn.clone();
        let mut residue = pdbtbx::Residue::new(resv, None, None).expect("Couldn't create residue");

        println!("ResidueNames: {}", res_name);
        let mut conformer =
            pdbtbx::Conformer::new(res_name, None, None).expect("Couldn't create Conformer");

        for atom in atoms {
            // coordinate vector is zero-indexed so we need to subtract 1
            let atom = &self.get_atom(atom.id - 1);
            conformer.add_atom(atom.clone());
        }
        residue.add_conformer(conformer);
        residue
    }
    pub fn create_chain(&self, chain: String) -> pdbtbx::Chain {
        let mut new_chain = pdbtbx::Chain::new(chain.clone()).unwrap();

        let residues: Vec<Residue> = self
            .get_residues_by_chain(chain.clone())
            .iter()
            .map(|res_num| self.create_residue(chain.clone(), *res_num))
            .collect();

        // index out of bounds: the len is 4557 but the index is 4557
        for res in residues {
            new_chain.add_residue(res.clone())
        }

        new_chain
    }
    /// Create a pdbtbx::PDB
    /// Note: Only handles the first Model....
    pub fn to_pdb(&self) -> PDB {
        // Create a Model. Need to fix this later if theres multiple models
        let mut model = pdbtbx::Model::new(1);
        let chains: Vec<pdbtbx::Chain> = self
            .get_chains()
            .iter()
            .map(|chainid| self.create_chain(chainid.to_string()))
            .collect();

        for chain in chains {
            model.add_chain(chain);
        }

        // Create PDB from Models
        let mut pdb = PDB::new();
        pdb.add_model(model);

        // Add Bonds Here
        for bond in &self.bond {
            // Todo: proper pymol bond--> pdbtbx bond
            pdb.add_bond(
                (bond.index_1 as usize, None),
                (bond.index_2 as usize, None),
                pdbtbx::Bond::Covalent,
            );
        }
        // Add Name/ Identifier
        let identifier = self.get_name().clone();
        pdb.identifier = Some(identifier);

        // Add Unit Cell and Symmetrey Info
        let (unit, sym) = self.get_unit_cell_symmetry();
        pdb.unit_cell = Some(unit);
        pdb.symmetry = Some(sym);

        pdb
    }
}

/// PymolSessionObjectData
///
/// Differentiate between serializing an object or a selection
/// They have different `shapes` and require custom deserialization.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum PymolSessionObjectData {
    PyObjectMolecule(PyObjectMolecule),
    SessionSelectorList(SessionSelectorList),
    // MolVariant(PyObjectMolecule),
    // SessionVariant(SessionSelector),
}
impl<'de> Deserialize<'de> for PymolSessionObjectData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match from_value::<PyObjectMolecule>(value.clone()) {
            Ok(molecule) => return Ok(PymolSessionObjectData::PyObjectMolecule(molecule)),
            Err(_) => {} // If it fails, we'll try the next option
        }
        if let Ok(selection) = from_value::<SessionSelectorList>(value.clone()) {
            return Ok(PymolSessionObjectData::SessionSelectorList(selection));
        }
        Err(serde::de::Error::custom("Failed to deserialize value"))
    }
}

/// pymol view of 25 floats is likely to be from the `SceneGetView`
///
/// [pymol](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer1/Scene.cpp#L872C1-L883C35)
///
/// /**
/// Get information required to define the geometry
/// of a particular view, for shipping to and from python
/// as a list of floats
/// @verbatim
/// 0-15 = 4x4 rotation matrix
/// 16-18 = position
/// 19-21 = origin
/// 22    = front plane
/// 23    = rear plane
/// 24    = orthoscopic flag
/// @endverbatim
/// @param[out] view buffer to fill
/// */
///
#[derive(Debug, Serialize)]
pub struct SceneView {
    pub rotation_matrix: [[f64; 4]; 4],
    pub position: [f64; 3],
    pub origin: [f64; 3],
    pub front_plane: f64,
    pub rear_plane: f64,
    pub orthoscopic_flag: f64,
}

impl SceneView {
    pub fn from_json_value(value: Value) -> Result<Self, serde_pickle::Error> {
        let array: [f64; 25] = serde_pickle::from_value(value)?;
        Ok(Self::from_array(array))
    }

    pub fn from_array(view: [f64; 25]) -> Self {
        SceneView {
            rotation_matrix: [
                [view[0], view[1], view[2], view[3]],
                [view[4], view[5], view[6], view[7]],
                [view[8], view[9], view[10], view[11]],
                [view[12], view[13], view[14], view[15]],
            ],
            position: [view[16], view[17], view[18]],
            origin: [view[19], view[20], view[21]],
            front_plane: view[22],
            rear_plane: view[23],
            orthoscopic_flag: view[24],
        }
    }
    pub fn to_array(&self) -> [f64; 25] {
        let mut array = [0.0; 25];
        for i in 0..4 {
            for j in 0..4 {
                array[i * 4 + j] = self.rotation_matrix[i][j];
            }
        }
        array[16..19].copy_from_slice(&self.position);
        array[19..22].copy_from_slice(&self.origin);
        array[22] = self.front_plane;
        array[23] = self.rear_plane;
        array[24] = self.orthoscopic_flag;
        array
    }
    pub fn get_location(&self) -> [f64; 9] {
        [
            self.rotation_matrix[0][0],
            self.rotation_matrix[0][1],
            self.rotation_matrix[0][2],
            self.rotation_matrix[1][0],
            self.rotation_matrix[1][1],
            self.rotation_matrix[1][2],
            self.rotation_matrix[2][0],
            self.rotation_matrix[2][1],
            self.rotation_matrix[2][2],
        ]
    }
    // pub fn get_translated_position(&self) -> [f64; 3] {
    //     // this is not right
    //     // use the 4x4 translation matrix and the position vector
    //     // to get a translated positions
    //     let mut translated = [0.0; 3];
    //     for i in 0..3 {
    //         translated[i] = self.rotation_matrix[i][0] * self.position[0]
    //             + self.rotation_matrix[i][1] * self.position[1]
    //             + self.rotation_matrix[i][2] * self.position[2]
    //             + self.rotation_matrix[i][3];
    //     }
    //     translated
    // }
    // pub fn get_translated_origin(&self) -> [f64; 3] {
    //     // this is not right
    //     let mut translated = [0.0; 3];
    //     for i in 0..3 {
    //         translated[i] = self.rotation_matrix[i][0] * self.origin[0]
    //             + self.rotation_matrix[i][1] * self.origin[1]
    //             + self.rotation_matrix[i][2] * self.origin[2]
    //             + self.rotation_matrix[i][3];
    //     }
    //     translated
    // }
}
impl<'de> Deserialize<'de> for SceneView {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let pickle_value = Value::deserialize(deserializer)?;

        if let Value::List(values) = pickle_value {
            if values.len() != 25 {
                return Err(serde::de::Error::custom("Expected 25 float values"));
            }

            let mut floats = [0.0; 25];
            for (i, value) in values.into_iter().enumerate() {
                if let Value::F64(f) = value {
                    floats[i] = f;
                } else {
                    return Err(serde::de::Error::custom("Expected float values"));
                }
            }

            Ok(SceneView {
                rotation_matrix: [
                    [floats[0], floats[1], floats[2], floats[3]],
                    [floats[4], floats[5], floats[6], floats[7]],
                    [floats[8], floats[9], floats[10], floats[11]],
                    [floats[12], floats[13], floats[14], floats[15]],
                ],
                position: [floats[16], floats[17], floats[18]],
                origin: [floats[19], floats[20], floats[21]],
                front_plane: floats[22],
                rear_plane: floats[23],
                orthoscopic_flag: floats[24],
            })
        } else {
            Err(serde::de::Error::custom("Expected a list of float values"))
        }
    }
}
/// SessionName
///
/// High level info about a session obeject.
/// Can be an object/molecule or a selection
///
#[derive(Debug, Deserialize, Serialize)]
pub struct SessionName {
    pub name: String,
    pub object: i32,
    pub visible: i32,
    unused: Option<bool>,
    unused2: i32,
    pub data: PymolSessionObjectData,
    pub group: String,
}

/// Session Selector
///
/// Defines a pymole seleciton in terms of names and
/// atom indices.
#[derive(Debug, Serialize)]
pub struct SessionSelector {
    // SelectorAsPyList
    // https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer3/Selector.cpp#L2926
    // list of lists
    // String: Name of the Object
    // Vec1: Index Object ( from VLA list )
    // Vec2: Tag Object ( from VLA list )
    // selector: Vec<(String, Vec<i32>, Vec<i32>)>, // this is there the selection bits are
    pub id: String,
    pub atom_index: Vec<i64>,
    pub atom_tag: Vec<i64>,
}
impl SessionSelector {
    pub fn to_component(&self) -> ComponentSelector {
        let mut expression_list: Vec<ComponentExpression> = vec![];
        for idx in &self.atom_index {
            let idx32: i32 = *idx as i32;
            expression_list.push(ComponentExpression {
                // internal representation of selection is available as the atom_index
                atom_index: Some(idx32),
                ..Default::default()
            });
        }
        ComponentSelector::ExpressionList(expression_list)
    }
}

// You might need a custom Deserialize implementation for SessionSelector
impl<'de> Deserialize<'de> for SessionSelector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (id, atom_index, atom_tag) = Deserialize::deserialize(deserializer)?;
        Ok(SessionSelector {
            id,
            atom_index,
            atom_tag,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SessionSelectorList(Vec<SessionSelector>);
impl SessionSelectorList {
    pub fn get_selectors(&self) -> &Vec<SessionSelector> {
        &self.0
    }
}

/// Global Pymol Settings
///
/// This enum is derived from [SettingInfo.h](https://github.com/schrodinger/pymol-open-source/blob/master/layer1/SettingInfo.h),
/// pymol's setting specification
///
#[derive(Serialize_repr, Deserialize_repr, PartialEq, Debug, Clone)]
#[repr(u32)]
pub enum SettingsEnum {
    BondingVdwCutoff = 0,
    MinMeshSpacing = 1,
    DotDensity = 2,
    DotMode = 3,
    SolventRadius = 4,
    SelCounter = 5,
    BgRgb = 6,
    Ambient = 7,
    Direct = 8,
    Reflect = 9,
    Light = 10,
    Power = 11,
    Antialias = 12,
    CavityCull = 13,
    GlAmbient = 14,
    SingleImage = 15,
    MovieDelay = 16,
    RibbonPower = 17,
    RibbonPowerB = 18,
    RibbonSampling = 19,
    RibbonRadius = 20,
    StickRadius = 21,
    HashMax = 22,
    Orthoscopic = 23,
    SpecReflect = 24,
    SpecPower = 25,
    SweepAngle = 26,
    SweepSpeed = 27,
    DotHydrogens = 28,
    DotRadius = 29,
    RayTraceFrames = 30,
    CacheFrames = 31,
    TrimDots = 32,
    CullSpheres = 33,
    Test1 = 34,
    Test2 = 35,
    SurfaceBest = 36,
    SurfaceNormal = 37,
    SurfaceQuality = 38,
    SurfaceProximity = 39,
    NormalWorkaround = 40,
    StereoAngle = 41,
    StereoShift = 42,
    LineSmooth = 43,
    LineWidth = 44,
    HalfBonds = 45,
    StickQuality = 46,
    StickOverlap = 47,
    StickNub = 48,
    AllStates = 49,
    Pickable = 50,
    AutoShowLines = 51,
    IdleDelay = 52,
    NoIdle = 53,
    FastIdle = 54,
    SlowIdle = 55,
    RockDelay = 56,
    DistCounter = 57,
    DashLength = 58,
    DashGap = 59,
    AutoZoom = 60,
    Overlay = 61,
    Text = 62,
    ButtonMode = 63,
    Valence = 64,
    NonbondedSize = 65,
    LabelColor = 66,
    RayTraceFog = 67,
    SpheroidScale = 68,
    RayTraceFogStart = 69,
    SpheroidSmooth = 70,
    SpheroidFill = 71,
    AutoShowNonbonded = 72,
    CacheDisplay = 73,
    MeshRadius = 74,
    BackfaceCull = 75,
    Gamma = 76,
    DotWidth = 77,
    AutoShowSelections = 78,
    AutoHideSelections = 79,
    SelectionWidth = 80,
    SelectionOverlay = 81,
    StaticSingletons = 82,
    // Unused83 = 83,
    DepthCue = 84,
    Specular = 85,
    Shininess = 86,
    SphereQuality = 87,
    Fog = 88,
    IsomeshAutoState = 89,
    MeshWidth = 90,
    CartoonSampling = 91,
    CartoonLoopRadius = 92,
    CartoonLoopQuality = 93,
    CartoonPower = 94,
    CartoonPowerB = 95,
    CartoonRectLength = 96,
    CartoonRectWidth = 97,
    InternalGuiWidth = 98,
    InternalGui = 99,
    CartoonOvalLength = 100,
    CartoonOvalWidth = 101,
    CartoonOvalQuality = 102,
    CartoonTubeRadius = 103,
    CartoonTubeQuality = 104,
    CartoonDebug = 105,
    RibbonWidth = 106,
    DashWidth = 107,
    DashRadius = 108,
    CgoRayWidthScale = 109,
    LineRadius = 110,
    CartoonRoundHelices = 111,
    CartoonRefineNormals = 112,
    CartoonFlatSheets = 113,
    CartoonSmoothLoops = 114,
    CartoonDumbbellLength = 115,
    CartoonDumbbellWidth = 116,
    CartoonDumbbellRadius = 117,
    CartoonFancyHelices = 118,
    CartoonFancySheets = 119,
    IgnorePdbSegi = 120,
    RibbonThrow = 121,
    CartoonThrow = 122,
    CartoonRefine = 123,
    CartoonRefineTips = 124,
    CartoonDiscreteColors = 125,
    NormalizeCcp4Maps = 126,
    SurfacePoor = 127,
    InternalFeedback = 128,
    CgoLineWidth = 129,
    CgoLineRadius = 130,
    Logging = 131,
    RobustLogs = 132,
    LogBoxSelections = 133,
    LogConformations = 134,
    ValenceSize = 135,
    SurfaceMiserable = 136,
    RayOpaqueBackground = 137,
    Transparency = 138,
    RayTexture = 139,
    RayTextureSettings = 140,
    SuspendUpdates = 141,
    FullScreen = 142,
    SurfaceMode = 143,
    SurfaceColor = 144,
    MeshMode = 145,
    MeshColor = 146,
    AutoIndicateFlags = 147,
    SurfaceDebug = 148,
    RayImproveShadows = 149,
    SmoothColorTriangle = 150,
    RayDefaultRenderer = 151,
    FieldOfView = 152,
    ReflectPower = 153,
    PreserveChempyIds = 154,
    SphereScale = 155,
    TwoSidedLighting = 156,
    SecondaryStructure = 157,
    AutoRemoveHydrogens = 158,
    RaiseExceptions = 159,
    StopOnExceptions = 160,
    Sculpting = 161,
    AutoSculpt = 162,
    SculptVdwScale = 163,
    SculptVdwScale14 = 164,
    SculptVdwWeight = 165,
    SculptVdwWeight14 = 166,
    SculptBondWeight = 167,
    SculptAnglWeight = 168,
    SculptPyraWeight = 169,
    SculptPlanWeight = 170,
    SculptingCycles = 171,
    SphereTransparency = 172,
    SphereColor = 173,
    SculptFieldMask = 174,
    SculptHbOverlap = 175,
    SculptHbOverlapBase = 176,
    LegacyVdwRadii = 177,
    SculptMemory = 178,
    ConnectMode = 179,
    CartoonCylindricalHelices = 180,
    CartoonHelixRadius = 181,
    ConnectCutoff = 182,
    SavePdbSs = 183,
    SculptLineWeight = 184,
    FitIterations = 185,
    FitTolerance = 186,
    BatchPrefix = 187,
    StereoMode = 188,
    CgoSphereQuality = 189,
    PdbLiteralNames = 190,
    WrapOutput = 191,
    FogStart = 192,
    State = 193,
    Frame = 194,
    RayShadow = 195,
    RibbonTraceAtoms = 196,
    Security = 197,
    StickTransparency = 198,
    RayTransparencyShadows = 199,
    SessionVersionCheck = 200,
    RayTransparencySpecular = 201,
    StereDoublePumpMono = 202,
    SphereSolvent = 203,
    MeshQuality = 204,
    MeshSolvent = 205,
    DotSolvent = 206,
    RayShadowFudge = 207,
    RayTriangleFudge = 208,
    DebugPick = 209,
    DotColor = 210,
    MouseLimit = 211,
    MouseScale = 212,
    TransparencyMode = 213,
    ClampColors = 214,
    PymolSpaceMaxRed = 215,
    PymolSpaceMaxGreen = 216,
    PymolSpaceMaxBlue = 217,
    PymolSpaceMinFactor = 218,
    RovingOrigin = 219,
    RovingLines = 220,
    RovingSticks = 221,
    RovingSpheres = 222,
    RovingLabels = 223,
    RovingDelay = 224,
    RovingSelection = 225,
    RovingByres = 226,
    RovingRibbon = 227,
    RovingCartoon = 228,
    RovingPolarContacts = 229,
    RovingPolarCutoff = 230,
    RovingNonbonded = 231,
    FloatLabels = 232,
    RovingDetail = 233,
    RovingNbSpheres = 234,
    RibbonColor = 235,
    CartoonColor = 236,
    RibbonSmooth = 237,
    AutoColor = 238,
    AutoColorNext = 239,
    RayInteriorColor = 240,
    CartoonHighlightColor = 241,
    CoulombUnitsFactor = 242,
    CoulombDielectric = 243,
    RayInteriorShadows = 244,
    RayInteriorTexture = 245,
    RovingMap1Name = 246,
    RovingMap2Name = 247,
    RovingMap3Name = 248,
    RovingMap1Level = 249,
    RovingMap2Level = 250,
    RovingMap3Level = 251,
    RovingIsomesh = 252,
    RovingIsosurface = 253,
    ScenesChanged = 254,
    GaussianBAdjust = 255,
    PdbStandardOrder = 256,
    CartoonSmoothFirst = 257,
    CartoonSmoothLast = 258,
    CartoonSmoothCycles = 259,
    CartoonFlatCycles = 260,
    MaxThreads = 261,
    ShowProgress = 262,
    UseDisplayLists = 263,
    CacheMemory = 264,
    SimplifyDisplayLists = 265,
    RetainOrder = 266,
    PdbHetatmSort = 267,
    PdbUseTerRecords = 268,
    CartoonTraceAtoms = 269,
    RayOversampleCutoff = 270,
    GaussianResolution = 271,
    GaussianBFloor = 272,
    SculptNbInterval = 273,
    SculptTorsWeight = 274,
    SculptTorsTolerance = 275,
    StickBall = 276,
    StickBallRatio = 277,
    StickFixedRadius = 278,
    CartoonTransparency = 279,
    DashRoundEnds = 280,
    HBondMaxAngle = 281,
    HBondCutoffCenter = 282,
    HBondCutoffEdge = 283,
    HBondPowerA = 284,
    HBondPowerB = 285,
    HBondCone = 286,
    SsHelixPsiTarget = 287,
    SsHelixPsiInclude = 288,
    SsHelixPsiExclude = 289,
    SsHelixPhiTarget = 290,
    SsHelixPhiInclude = 291,
    SsHelixPhiExclude = 292,
    SsStrandPsiTarget = 293,
    SsStrandPsiInclude = 294,
    SsStrandPsiExclude = 295,
    SsStrandPhiTarget = 296,
    SsStrandPhiInclude = 297,
    SsStrandPhiExclude = 298,
    MovieLoop = 299,
    PdbRetainIds = 300,
    PdbNoEndRecord = 301,
    CgoDotWidth = 302,
    CgoDotRadius = 303,
    DeferUpdates = 304,
    NormalizeOMaps = 305,
    SwapDsn6Bytes = 306,
    PdbInsertionsGoFirst = 307,
    RovingOriginZ = 308,
    RovingOriginZCushion = 309,
    SpecularIntensity = 310,
    OverlayLines = 311,
    RayTransparencySpecCut = 312,
    InternalPrompt = 313,
    NormalizeGrdMaps = 314,
    RayBlendColors = 315,
    RayBlendRed = 316,
    RayBlendGreen = 317,
    RayBlendBlue = 318,
    PngScreenGamma = 319,
    PngFileGamma = 320,
    EditorLabelFragments = 321,
    InternalGuiControlSize = 322,
    AutoDss = 323,
    TransparencyPickingMode = 324,
    VirtualTrackball = 325,
    PdbReformatNamesMode = 326,
    RayPixelScale = 327,
    LabelFontId = 328,
    PdbConectAll = 329,
    ButtonModeName = 330,
    SurfaceType = 331,
    DotNormals = 332,
    SessionMigration = 333,
    MeshNormals = 334,
    MeshType = 335,
    DotLighting = 336,
    MeshLighting = 337,
    SurfaceSolvent = 338,
    TriangleMaxPasses = 339,
    RayInteriorReflect = 340,
    InternalGuiMode = 341,
    SurfaceCarveSelection = 342,
    SurfaceCarveState = 343,
    SurfaceCarveCutoff = 344,
    SurfaceClearSelection = 345,
    SurfaceClearState = 346,
    SurfaceClearCutoff = 347,
    SurfaceTrimCutoff = 348,
    SurfaceTrimFactor = 349,
    RayMaxPasses = 350,
    ActiveSelections = 351,
    RayTransparencyContrast = 352,
    SeqView = 353,
    MouseSelectionMode = 354,
    SeqViewLabelSpacing = 355,
    SeqViewLabelStart = 356,
    SeqViewFormat = 357,
    SeqViewLocation = 358,
    SeqViewOverlay = 359,
    AutoClassifyAtoms = 360,
    CartoonNucleicAcidMode = 361,
    SeqViewColor = 362,
    SeqViewLabelMode = 363,
    SurfaceRampAboveMode = 364,
    Stereo = 365,
    WizardPromptMode = 366,
    CoulombCutoff = 367,
    SliceTrackCamera = 368,
    SliceHeightScale = 369,
    SliceHeightMap = 370,
    SliceGrid = 371,
    SliceDynamicGrid = 372,
    SliceDynamicGridResolution = 373,
    PdbInsureOrthogonal = 374,
    RayDirectShade = 375,
    StickColor = 376,
    CartoonPuttyRadius = 377,
    CartoonPuttyQuality = 378,
    CartoonPuttyScaleMin = 379,
    CartoonPuttyScaleMax = 380,
    CartoonPuttyScalePower = 381,
    CartoonPuttyRange = 382,
    CartoonSideChainHelper = 383,
    SurfaceOptimizeSubsets = 384,
    Multiplex = 385,
    TextureFonts = 386,
    PqrNoChainId = 387,
    Animation = 388,
    AnimationDuration = 389,
    SceneAnimation = 390,
    LineStickHelper = 391,
    RayOrthoscopic = 392,
    RibbonSideChainHelper = 393,
    SelectionWidthMax = 394,
    SelectionWidthScale = 395,
    SceneCurrentName = 396,
    Presentation = 397,
    PresentationMode = 398,
    PdbTruncateResidueName = 399,
    SceneLoop = 400,
    SweepMode = 401,
    SweepPhase = 402,
    SceneRestartMovieDelay = 403,
    MouseRestartMovieDelay = 404,
    AngleSize = 405,
    AngleLabelPosition = 406,
    DihedralSize = 407,
    DihedralLabelPosition = 408,
    DeferBuildsMode = 409,
    SeqViewDiscreteByState = 410,
    SceneAnimationDuration = 411,
    Wildcard = 412,
    AtomNameWildcard = 413,
    IgnoreCase = 414,
    PresentationAutoQuit = 415,
    EditorAutoDihedral = 416,
    PresentationAutoStart = 417,
    ValidateObjectNames = 418,
    UnusedBooleanDefTrue = 419,
    AutoShowSpheres = 420,
    SphereMode = 421,
    SpherePointMaxSize = 422,
    SpherePointSize = 423,
    PdbHonorModelNumber = 424,
    RankAssistedSorts = 425,
    RibbonNucleicAcidMode = 426,
    CartoonRingMode = 427,
    CartoonRingWidth = 428,
    CartoonRingColor = 429,
    CartoonRingFinder = 430,
    CartoonTubeCap = 431,
    CartoonLoopCap = 432,
    NvidiaBugs = 433,
    ImageDotsPerInch = 434,
    OpaqueBackground = 435,
    DrawFrames = 436,
    ShowAlphaChecker = 437,
    MatrixMode = 438,
    EditorAutoOrigin = 439,
    SessionFile = 440,
    CgoTransparency = 441,
    LegacyMouseZoom = 442,
    AutoNumberSelections = 443,
    SculptVdwVisMode = 444,
    SculptVdwVisMin = 445,
    SculptVdwVisMid = 446,
    SculptVdwVisMax = 447,
    CartoonLadderMode = 448,
    CartoonLadderRadius = 449,
    CartoonLadderColor = 450,
    CartoonNucleicAcidColor = 451,
    CartoonRingTransparency = 452,
    LabelSize = 453,
    SpecDirect = 454,
    LightCount = 455,
    Light2 = 456,
    Light3 = 457,
    HideUnderscoreNames = 458,
    SelectionRoundPoints = 459,
    DistanceExclusion = 460,
    HBondExclusion = 461,
    LabelShadowMode = 462,
    Light4 = 463,
    Light5 = 464,
    Light6 = 465,
    Light7 = 466,
    LabelOutlineColor = 467,
    RayTraceMode = 468,
    RayTraceGain = 469,
    SelectionVisibleOnly = 470,
    LabelPosition = 471,
    RayTraceDepthFactor = 472,
    RayTraceSlopeFactor = 473,
    RayTraceDiscoFactor = 474,
    RayShadowDecayFactor = 475,
    RayInteriorMode = 476,
    RayLegacyLighting = 477,
    SculptAutoCenter = 478,
    PdbDiscreteChains = 479,
    PdbUnbondCations = 480,
    SculptTriScale = 481,
    SculptTriWeight = 482,
    SculptTriMin = 483,
    SculptTriMax = 484,
    SculptTriMode = 485,
    PdbEchoTags = 486,
    ConnectBonded = 487,
    SpecDirectPower = 488,
    Light8 = 489,
    Light9 = 490,
    RayShadowDecayRange = 491,
    SpecCount = 492,
    SculptMinScale = 493,
    SculptMinWeight = 494,
    SculptMinMin = 495,
    SculptMinMax = 496,
    SculptMaxScale = 497,
    SculptMaxWeight = 498,
    SculptMaxMin = 499,
    SculptMaxMax = 500,
    SurfaceCircumscribe = 501,
    SculptAvdWeight = 502,
    SculptAvdGap = 503,
    SculptAvdRange = 504,
    SculptAvdExcl = 505,
    AsyncBuilds = 506,
    FetchPath = 507,
    CartoonRingRadius = 508,
    RayColorRamps = 509,
    RayHintCamera = 510,
    RayHintShadow = 511,
    StickValenceScale = 512,
    SeqViewAlignment = 513,
    SeqViewUnalignedMode = 514,
    SeqViewUnalignedColor = 515,
    SeqViewFillChar = 516,
    SeqViewFillColor = 517,
    SeqViewLabelColor = 518,
    SurfaceCarveNormalCutoff = 519,
    TraceAtomsMode = 520,
    SessionChanged = 521,
    RayClipShadows = 522,
    MouseWheelScale = 523,
    NonbondedTransparency = 524,
    RaySpecLocal = 525,
    LineColor = 526,
    RayLabelSpecular = 527,
    MeshSkip = 528,
    LabelDigits = 529,
    LabelDistanceDigits = 530,
    LabelAngleDigits = 531,
    LabelDihedralDigits = 532,
    SurfaceNegativeVisible = 533,
    SurfaceNegativeColor = 534,
    MeshNegativeVisible = 535,
    MeshNegativeColor = 536,
    GroupAutoMode = 537,
    GroupFullMemberNames = 538,
    GradientMaxLength = 539,
    GradientMinLength = 540,
    GradientMinSlope = 541,
    GradientNormalMinDot = 542,
    GradientStepSize = 543,
    GradientSpacing = 544,
    GradientSymmetry = 545,
    RayTraceColor = 546,
    GroupArrowPrefix = 547,
    SuppressHidden = 548,
    SessionCompression = 549,
    MovieFps = 550,
    RayTransparencyOblique = 551,
    RayTraceTransCutoff = 552,
    RayTracePersistCutoff = 553,
    RayTransparencyObliquePower = 554,
    RayScatter = 555,
    HBondFromProton = 556,
    AutoCopyImages = 557,
    MoeSeparateChains = 558,
    TransparencyGlobalSort = 559,
    HideLongBonds = 560,
    AutoRenameDuplicateObjects = 561,
    PdbHetatmGuessValences = 562,
    EllipsoidQuality = 563,
    CgoEllipsoidQuality = 564,
    MovieAnimateByFrame = 565,
    RampBlendNearbyColors = 566,
    AutoDeferBuilds = 567,
    EllipsoidProbability = 568,
    EllipsoidScale = 569,
    EllipsoidColor = 570,
    EllipsoidTransparency = 571,
    MovieRock = 572,
    CacheMode = 573,
    DashColor = 574,
    AngleColor = 575,
    DihedralColor = 576,
    GridMode = 577,
    CacheMax = 578,
    GridSlot = 579,
    GridMax = 580,
    CartoonPuttyTransform = 581,
    Rock = 582,
    ConeQuality = 583,
    PdbFormalCharges = 584,
    AtiBugs = 585,
    GeometryExportMode = 586,
    MouseGrid = 587,
    MeshCutoff = 588,
    MeshCarveSelection = 589,
    MeshCarveState = 590,
    MeshCarveCutoff = 591,
    MeshClearSelection = 592,
    MeshClearState = 593,
    MeshClearCutoff = 594,
    MeshGridMax = 595,
    SessionCacheOptimize = 596,
    SdofDragScale = 597,
    SceneButtonsMode = 598,
    SceneButtons = 599,
    MapAutoExpandSym = 600,
    ImageCopyAlways = 601,
    MaxUps = 602,
    AutoOverlay = 603,
    StickBallColor = 604,
    StickHScale = 605,
    SculptPyraInvWeight = 606,
    KeepAlive = 607,
    FitKabsch = 608,
    StereoDynamicStrength = 609,
    DynamicWidth = 610,
    DynamicWidthFactor = 611,
    DynamicWidthMin = 612,
    DynamicWidthMax = 613,
    DrawMode = 614,
    CleanElectroMode = 615,
    ValenceMode = 616,
    ShowFrameRate = 617,
    MoviePanel = 618,
    MouseZScale = 619,
    MovieAutoStore = 620,
    MovieAutoInterpolate = 621,
    MoviePanelRowHeight = 622,
    SceneFrameMode = 623,
    SurfaceCavityMode = 624,
    SurfaceCavityRadius = 625,
    SurfaceCavityCutoff = 626,
    MotionPower = 627,
    MotionBias = 628,
    MotionSimple = 629,
    MotionLinear = 630,
    MotionHand = 631,
    PdbIgnoreConect = 632,
    EditorBondCycleMode = 633,
    MovieQuality = 634,
    LabelAnchor = 635,
    FetchHost = 636,
    DynamicMeasures = 637,
    NeighborCutoff = 638,
    HeavyNeighborCutoff = 639,
    PolarNeighborCutoff = 640,
    SurfaceResidueCutoff = 641,
    SurfaceUseShader = 642,
    CartoonUseShader = 643,
    StickUseShader = 644,
    LineUseShader = 645,
    SphereUseShader = 646,
    UseShaders = 647,
    ShadersFromDisk = 648,
    VolumeBitDepth = 649,
    VolumeColor = 650,
    VolumeLayers = 651,
    VolumeDataRange = 652,
    AutoDeferAtomCount = 653,
    DefaultRefmacNames = 654,
    DefaultPhenixNames = 655,
    DefaultPhenixNoFillNames = 656,
    DefaultBusterNames = 657,
    DefaultFofcMapRep = 658,
    Default2fofcMapRep = 659,
    AtomTypeFormat = 660,
    AutocloseDialogs = 661,
    BgGradient = 662,
    BgRgbTop = 663,
    BgRgbBottom = 664,
    RayVolume = 665,
    RibbonTransparency = 666,
    StateCounterMode = 667,
    CgoUseShader = 668,
    CgoShaderUbColor = 669,
    CgoShaderUbNormal = 670,
    CgoLighting = 671,
    MeshUseShader = 672,
    StickDebug = 673,
    CgoDebug = 674,
    StickRoundNub = 675,
    StickGoodGeometry = 676,
    StickAsCylinders = 677,
    MeshAsCylinders = 678,
    LineAsCylinders = 679,
    RibbonAsCylinders = 680,
    RibbonUseShader = 681,
    ExclDisplayListsShaders = 682,
    DashUseShader = 683,
    DashAsCylinders = 684,
    NonbondedUseShader = 685,
    NonbondedAsCylinders = 686,
    CylindersShaderFilterFaces = 687,
    NbSpheresSize = 688,
    NbSpheresQuality = 689,
    NbSpheresUseShader = 690,
    RenderAsCylinders = 691,
    AlignmentAsCylinders = 692,
    CartoonNucleicAcidAsCylinders = 693,
    CgoShaderUbFlags = 694,
    AntialiasShader = 695,
    OffscreenRenderingMultiplier = 696,
    CylinderShaderFfWorkaround = 697,
    SurfaceColorSmoothing = 698,
    SurfaceColorSmoothingThreshold = 699,
    DotUseShader = 700,
    DotAsSheres = 701,
    AmbientOcclusionMode = 702,
    AmbientOcclusionScale = 703,
    AmbientOcclusionSmooth = 704,
    SmoothHalfBonds = 705,
    AnaglyphMode = 706,
    EditLight = 707,
    SuspendUndo = 708,
    SuspendUndoAtomCount = 709,
    SuspendDeferred = 710,
    PickSurface = 711,
    BgImageFilename = 712,
    BgImageMode = 713,
    BgImageTilesize = 714,
    BgImageLinear = 715,
    LoadObjectPropsDefault = 716,
    LoadAtomPropsDefault = 717,
    LabelPlacementOffset = 718,
    PdbConectNodup = 719,
    LabelConnector = 720,
    LabelConnectorMode = 721,
    LabelConnectorColor = 722,
    LabelConnectorWidth = 723,
    LabelConnectorExtLength = 724,
    LabelBgColor = 725,
    UseGeometryShaders = 726,
    LabelRelativeMode = 727,
    LabelScreenPoint = 728,
    LabelMultilineSpacing = 729,
    LabelMultilineJustification = 730,
    LabelPadding = 731,
    LabelBgTransparency = 732,
    LabelBgOutline = 733,
    RayLabelConnectorFlat = 734,
    DashTransparency = 735,
    PickLabels = 736,
    LabelZTarget = 737,
    SessionEmbedsData = 738,
    VolumeMode = 739,
    Trilines = 740,
    ColladaExportLighting = 741,
    ColladaGeometryMode = 742,
    PrecomputedLighting = 743,
    Chromadepth = 744,
    PseExportVersion = 745,
    CifUseAuth = 746,
    Assembly = 747,
    CifKeepinmemory = 748,
    PseBinaryDump = 749,
    CartoonGapCutoff = 750,
    IgnoreCaseChain = 751,
    ValenceZeroScale = 752,
    ValenceZeroMode = 753,
    AutoShowClassified = 754,
    ColladaBackgroundBox = 755,
    Pick32bit = 756,
    CartoonAllAlt = 757,
    DisplayScaleFactor = 758,
    PickShading = 759,
    FetchTypeDefault = 760,
    EditorAutoMeasure = 761,
    SurfaceSmoothEdges = 762,
    ChemCompCartnUse = 763,
    ColoredFeedback = 764,
    SdfWriteZeroOrderBonds = 765,
    CifMetalcAsZeroOrderBonds = 766,
    SeqViewGapMode = 767,
    InternalGuiNameColorMode = 768,
    OpenvrGuiFov = 769,
    OpenvrGuiAlpha = 770,
    OpenvrGuiUseAlpha = 771,
    OpenvrGuiSceneColor = 772,
    OpenvrGuiSceneAlpha = 773,
    OpenvrGuiBackColor = 774,
    OpenvrGuiBackAlpha = 775,
    OpenvrGuiUseBackdrop = 776,
    OpenvrGuiOverlay = 777,
    OpenvrGuiText = 778,
    OpenvrDisableClipping = 779,
    OpenvrNearPlane = 780,
    OpenvrFarPlane = 781,
    OpenvrCutLaser = 782,
    OpenvrLaserWidth = 783,
    OpenvrGuiDistance = 784,
    CartoonSmoothCylinderCycles = 785,
    CartoonSmoothCylinderWindow = 786,
    IsosurfaceAlgorithm = 787,
    CellCentered = 788,
    HalogenBondDistance = 789,
    HalogenBondAsDonorMinDonorAngle = 790,
    HalogenBondAsDonorMinAcceptorAngle = 791,
    HalogenBondAsAcceptorMinDonorAngle = 792,
    HalogenBondAsAcceptorMinAcceptorAngle = 793,
    HalogenBondAsAcceptorMaxAcceptorAngle = 794,
    SaltBridgeDistance = 795,
    UseTessellationShaders = 796,
    CellColor = 797,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Settings {
    pub setting: SettingsEnum,
    pub label: i32,
    pub value: CustomValue,
}

fn int_to_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    match u8::deserialize(deserializer)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(Error::custom(format!("Invalid boolean value: {}", other))),
    }
}

// Todo:
//
// struct PyObjectGadget {}
// struct PyObjectDist {}
// struct PyObjectMap {}
// struct PyObjectMesh {}
// struct PyObjectSurface {}
// struct PyObjectCGO {}
// struct PyObjectAlignment {}
// struct PyObjectGroup {}
// struct PyObjectVolume {}
// struct PyObjectCallback {}
// struct PyObjectCurve {}
