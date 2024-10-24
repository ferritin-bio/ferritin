//! # PSEData
//!
//! PSEData is a struct for loading and serializing pymol PSE data.
//!
//! Currently the parsers are working for small test cases of molecules and selections. Additional parser structs would be required for
//! other PSE data types which include the folloing:
//!

use crate::molviewspec::nodes::{self as mvsnodes, ColorNamesT, State};
use crate::pymolparsing::parsing::{
    PyObjectMolecule, PymolSessionObjectData, SceneView, SessionName, SessionSelectorList,
    Settings, SettingsEnum,
};
use pdbtbx::PDB;
use serde::{Deserialize, Serialize};
use serde_pickle::de::{from_reader, DeOptions};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// PSEData represents the structure of a PyMOL Session File (PSE).
///
/// We lean heavily on `serde_pickle` to deserialize the PSE binary
/// file into named structs, of which `PSEData` is the highlest level object containing
/// most of the required methods for operating on PSE files.
///
///  This struct contains various components of a PyMOL session, including:
/// - Version information
/// - Color settings
/// - View settings
/// - Movie scenes
/// - Custom settings
/// - Cached data
/// - Session names and associated objects
///
/// To be implemented in time:
///
/// PyObject :
/// -[pyobject](https://github.com/schrodinger/pymol-open-source/blob/03d7a7fcf0bd95cd93d710a1268dbace2ed77765/layer1/PyMOLObject.cpp#L681)
///
/// Python Obects:
/// - Object
/// - Gadget
/// - Molecule    ---> WIP.
/// - Dist
/// - Map
/// - Mesh
/// - Slice
/// - Surface
/// - CGO
/// - Alignment
/// - Group
/// - Volume
/// - Callback
/// - Curve
/// - Selection ---> WIP.
///
#[derive(Debug, Deserialize, Serialize)]
pub struct PSEData {
    pub version: i32,
    pub main: Vec<i64>,
    pub colors: Vec<i32>,
    pub color_ext: Vec<i32>,
    pub unique_settings: Vec<i32>,
    pub selector_secrets: Vec<i32>,
    pub editor: Vec<i32>,
    pub view: SceneView,
    pub view_dict: HashMap<String, String>,
    #[serde(with = "serde_bytes")]
    pub wizard: Vec<u8>,
    pub moviescenes: Vec<Vec<i32>>,
    // High level state settings: we need to prpogate these..
    pub settings: Vec<Settings>,
    pub movie: (
        i32,
        i32,
        Vec<f32>,
        i32,
        Option<bool>, // this will probably need to be modified
        Option<bool>,
        Option<bool>,
    ),
    // not needed?
    // session: HashMap<String, Value>,
    pub cache: Vec<usize>,
    // name is the trickiest bit
    pub names: Vec<Option<SessionName>>,
}

impl PSEData {
    /// Load pymole `.pse` binary file to struct.
    pub fn load(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(file_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let options = DeOptions::new()
            .replace_unresolved_globals()
            .decode_strings();
        let pse_data_vals: serde_pickle::Value = from_reader(&buffer[..], options).unwrap();
        let pse_data: PSEData = serde_pickle::from_value(pse_data_vals).unwrap();
        Ok(pse_data)
    }
    /// Serialize to JSON
    pub fn to_json(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(file_path, json)?;
        Ok(())
    }
    // adds custom colors to auto colors to get the index of colors
    pub fn get_full_colorlist() {
        // https://github.com/schrodinger/pymol-open-source/blob/master/layer1/Color.cpp#L415
        unimplemented!()
    }

    /// session is where all the action happens
    pub fn get_session_names(&self) -> Vec<String> {
        self.names
            .iter()
            .filter_map(|session_name| {
                session_name
                    .as_ref()
                    .map(|session| session.name.to_string())
            })
            .collect()
    }
    /// Global Pymol Settings
    pub fn get_setting(&self, setting: SettingsEnum) -> Option<Settings> {
        self.settings.iter().find(|s| s.setting == setting).cloned()
    }
    // pub fn get_view(&self) -> &Vec<f32> {
    //     let view = self.view; // f32: 25
    // }
    pub fn get_molecule_data(&self) -> Vec<&PyObjectMolecule> {
        self.names
            .iter()
            .filter_map(|session_name| session_name.as_ref())
            .filter_map(|session| match &session.data {
                PymolSessionObjectData::PyObjectMolecule(a) => Some(a),
                _ => None,
            })
            .collect()
    }

    pub fn get_selection_data(&self) -> Vec<&SessionSelectorList> {
        self.names
            .iter()
            .filter_map(|session_name| session_name.as_ref())
            .filter_map(|session| match &session.data {
                PymolSessionObjectData::SessionSelectorList(a) => Some(a),
                _ => None,
            })
            .collect()
    }

    pub fn get_location_as_transform(&self) -> mvsnodes::TransformParams {
        let location = self.view.get_location();
        mvsnodes::TransformParams {
            rotation: Some(location.to_vec()),
            ..Default::default()
        }
    }
    /// Convert Pymol to pdbtbx::PDB
    pub fn create_pdb(&self) -> PDB {
        // todo: extend this to more than one molecuelo and/or to modify the global scene
        let moldata = &self.get_molecule_data();
        let first_mol = moldata[0];
        first_mol.to_pdb()
    }
    /// Pymol --> PDBTBX --> PDB --> file
    pub fn save_pdbs(&self, file_path: &str) -> std::io::Result<()> {
        let path = std::path::Path::new(file_path);
        let pdb_folder = path.join("pdb");
        std::fs::create_dir_all(&pdb_folder)?;
        let mut file_list = Vec::new();
        for molecule in self.get_molecule_data().iter() {
            let pdb = molecule.to_pdb();
            let filename = format!("{}.pdb", molecule.get_name());
            let file_path = pdb_folder.join(&filename);
            let _ = pdbtbx::save_pdb(
                &pdb,
                file_path.to_str().expect("Invalid UTF-8 in file path"),
                pdbtbx::StrictnessLevel::Strict,
            );
            file_list.push(filename);
        }
        let contents = file_list.join("\n");
        std::fs::write(path.join("pdb_contents.txt"), contents)?;
        Ok(())
    }
    /// Convert to MSVJ Formay.
    /// see also: [mol-view-spec](https://github.com/molstar/mol-view-spec)
    pub fn create_molviewspec(&self) -> State {
        // write state for loading the PDB files
        let mut state = State::new();

        // Add Global Camera Data
        // let [o1, o2, o3] = self.view.origin;
        // let [p1, p2, p3] = self.view.position;
        //
        // let [o1, o2, o3] = self.view.get_translated_origin();
        // let [p1, p2, p3] = self.view.get_translated_position();
        //
        // let camparam = CameraParams {
        //     // https://molstar.org/mol-view-spec-docs/camera-settings/
        //     target: (o1, o2, o3), // <--- Todo
        //     position: (p1, p2, p3),
        //     ..Default::default() // <--- Todo
        // };
        // state.camera(camparam);

        // It will be easier to set the focus based on all of the components in the PDB then trying to match pymol exactly
        // let focus = FocusInlineParams {};

        // Add Molecule Data
        for molecule in self.get_molecule_data() {
            let molname = molecule.get_name();

            let structure = state
                .download(&format!("pdb/{}.pdb", molname))
                .expect("Create a Download node with a URL")
                .parse(mvsnodes::ParseParams {
                    format: mvsnodes::ParseFormatT::Pdb,
                })
                .expect("Parseable option")
                .assembly_structure(mvsnodes::StructureParams {
                    structure_type: mvsnodes::StructureTypeT::Model,
                    ..Default::default()
                })
                .expect("a set of Structure options");

            // add base structure component then any selections that may be relevant
            structure
                .component(mvsnodes::ComponentSelector::default())
                .expect("defined a valid component")
                .representation(mvsnodes::RepresentationTypeT::Cartoon);

            // this works great
            let transform = self.get_location_as_transform();
            structure.transform(transform);

            // selections return MVD ComponentExpression
            let selection_data = self.get_selection_data()[0];
            for selector in selection_data.get_selectors() {
                if selector.id == molname {
                    println!("Found a selction for Model {}!!!!", molname);
                    let component = selector.to_component();
                    structure
                        .component(component)
                        .expect("defined a valid component")
                        .representation(mvsnodes::RepresentationTypeT::BallAndStick)
                        .expect("a representation")
                        // to do add colors and other settings....
                        .color(
                            mvsnodes::ColorT::Named(ColorNamesT::Magenta),
                            mvsnodes::ComponentSelector::default(),
                        )
                        .expect("a color");
                }
            }
        }

        state
    }
    ///  Pymol --> MSVJ --? disk
    pub fn to_disk(&self, file_path: &str) -> std::io::Result<()> {
        let path = std::path::Path::new(file_path);
        let msvj_file = path.join("state.mvsj");
        let state = self.create_molviewspec();
        let pretty_json = serde_json::to_string_pretty(&state)?;
        self.save_pdbs(file_path)?;
        std::fs::write(msvj_file, pretty_json)?;
        Ok(())
    }
    /// this one will write  a ready-to-go folder with pdbs, an msvj file, and the
    /// html/css/js needed to load them
    pub fn to_disk_full(&self, file_path: &str) -> std::io::Result<()> {
        // Create the directory if it doesn't exist
        fs::create_dir_all(file_path)?;

        // Copy our standard files
        let resources_dir = Path::new("resources");
        let files_to_copy = ["index.html", "molstar.css", "molstar.js"];
        for file_name in files_to_copy.iter() {
            let src_path = resources_dir.join(file_name);
            let dest_path = Path::new(file_path).join(file_name);
            fs::copy(&src_path, &dest_path)?;
        }
        // copy our custom files
        let _ = self.to_disk(file_path);
        Ok(())
    }
    /// To loadable MSVJ URL
    /// See also [msvj URL encoding](https://molstar.org/mol-view-spec-docs/mvs-molstar-extension/)
    pub fn to_mvsj_url(&self) -> String {
        let state = self.create_molviewspec();
        state.to_url()
    }
}
