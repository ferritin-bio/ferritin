//! This module has functionality to build up MolViewSpec.
//!
//! MolViewSpec (MVS) is a lightweight, JSON-based specification for describing molecular structures and their visual representation.
//! It is designed to be human-readable, easily shareable, and compatible with various molecular visualization tools.
//!
//! - MolViewSpec GitHub repository: https://github.com/molstar/mol-view-spec
//! - MolViewSpec documentation: https://molstar.org/viewer/molviewspec/
//! - MolStar Viewer (which supports MVS): https://molstar.org/viewer/
//!
//! We try to adhere very closely to the python library API. Almost all of the action happens
//! on the `Nodes`. Because we are building a nested tree of data, we need most of the parts to
//! be mutable and you can see almost all of the function calls:
//!
//! 1. check the type and the params and act only on correct parent nodes
//! 2. beacuse of 1, we return `Option<>`
//! 3. If a node is returned is it `&mut Node`
//!
//!
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json;
use urlencoding;
use validator::Validate;

// KindT
//
// Enum of node types corresponding to the MolViewSpec Nodes
//
#[derive(PartialEq, Serialize, Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "snake_case")]
pub enum KindT {
    #[default]
    Root,
    Camera,
    Canvas,
    Color,
    ColorFromSource,
    ColorFromUri,
    Component,
    ComponentFromSource,
    ComponentFromUri,
    Download,
    Focus,
    GenericVisuals,
    Label,
    LabelFromSource,
    LabelFromUri,
    Line,
    Parse,
    Representation,
    Sphere,
    Structure,
    Tooltip,
    TooltipFromSource,
    TooltipFromUri,
    Transform,
}

// NodeParams
//
// Enum of params per node type. Each of the variants are typed.
//
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum NodeParams {
    DownloadParams(DownloadParams),
    ParseParams(ParseParams),
    StructureParams(StructureParams),
    RepresentationParams(RepresentationParams),
    DataFromUriParams(DataFromUriParams),
    DataFromSourceParams(DataFromSourceParams),
    ComponentInlineParams(ComponentInlineParams),
    ComponentFromUriParams(ComponentFromUriParams),
    ComponentFromSourceParams(ComponentFromSourceParams),
    ColorInlineParams(ColorInlineParams),
    ColorFromUriParams(ColorFromUriParams),
    ColorFromSourceParams(ColorFromSourceParams),
    LabelInlineParams(LabelInlineParams),
    LabelFromUriParams(LabelFromUriParams),
    LabelFromSourceParams(LabelFromSourceParams),
    TooltipInlineParams(TooltipInlineParams),
    TooltipFromUriParams(TooltipFromUriParams),
    TooltipFromSourceParams(TooltipFromSourceParams),
    FocusInlineParams(FocusInlineParams),
    TransformParams(TransformParams),
    CameraParams(CameraParams),
    CanvasParams(CanvasParams),
    SphereParams(SphereParams),
    LineParams(LineParams),
}

/// Node
///
/// This is the core datastructure for generating MSVJ files. Each node type can have a type, params, and children.
///
/// Methods derived from the Python API found [here](https://github.com/molstar/mol-view-spec/blob/master/molviewspec/molviewspec/builder.py)
///
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Node {
    pub kind: KindT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<NodeParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<Node>>,
}
impl Node {
    // Common to All Nodes
    pub fn new(kind: KindT, params: Option<NodeParams>) -> Node {
        Node {
            kind,
            params,
            children: None,
        }
    }
    pub fn add_child(&mut self, node: Node) {
        match &mut self.children {
            Some(children) => children.push(node),
            None => self.children = Some(vec![node]),
        }
    }
    pub fn get_kind(&self) -> &KindT {
        &self.kind
    }
    /// Create the download node
    pub fn download(&mut self, url: &str) -> Option<&mut Node> {
        if self.kind == KindT::Root {
            let url = url.to_string();
            let download_node = Node::new(
                KindT::Download,
                Some(NodeParams::DownloadParams(DownloadParams { url })),
            );

            self.children
                .get_or_insert_with(Vec::new)
                .push(download_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }
    /// Parse a Download Node
    pub fn parse(&mut self, params: ParseParams) -> Option<&mut Node> {
        println!("In the parse node!");
        println!("{:?}", self.kind);
        if self.kind == KindT::Download {
            let parse_node = Node::new(KindT::Parse, Some(NodeParams::ParseParams(params)));
            self.children.get_or_insert_with(Vec::new).push(parse_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }

    // Parse methods ------------------------------------------------------

    /// Create a structure for the deposited coordinates.
    ///  :param model_index: 0-based model index in case multiple NMR frames are present
    /// :param block_index: 0-based block index in case multiple mmCIF or SDF data blocks are present
    /// :param block_header: Reference a specific mmCIF or SDF data block by its block header
    /// :return: a builder that handles operations at structure level
    pub fn model_structure() {
        unimplemented!()
    }
    /// Parse a Download Node
    pub fn assembly_structure(&mut self, params: StructureParams) -> Option<&mut Node> {
        if self.kind == KindT::Parse {
            let struct_node =
                Node::new(KindT::Structure, Some(NodeParams::StructureParams(params)));
            self.children.get_or_insert_with(Vec::new).push(struct_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }
    /// Symmetry Structure
    pub fn symmetry_structure(&mut self, params: StructureParams) -> Option<&mut Node> {
        // todo: this is the same as the regular structure bit above......
        if self.kind == KindT::Parse {
            let struct_node =
                Node::new(KindT::Structure, Some(NodeParams::StructureParams(params)));
            self.children.get_or_insert_with(Vec::new).push(struct_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }

    /// Parse a Download Node
    pub fn symmetry_mates_structure(&mut self, params: StructureParams) -> Option<&mut Node> {
        // todo: this is the same as the regular structure bit above......
        if self.kind == KindT::Parse {
            let struct_node =
                Node::new(KindT::Structure, Some(NodeParams::StructureParams(params)));
            self.children.get_or_insert_with(Vec::new).push(struct_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }

    // Structure methods ------------------------------------------------------

    /// Create a Component
    pub fn component(&mut self, selector: ComponentSelector) -> Option<&mut Node> {
        if self.kind == KindT::Structure {
            let component_node = match selector {
                ComponentSelector::Selector(sel) => Node::new(
                    KindT::Component,
                    Some(NodeParams::ComponentInlineParams(ComponentInlineParams {
                        selector: ComponentSelector::Selector(sel),
                    })),
                ),
                ComponentSelector::Expression(expr) => Node::new(
                    KindT::Component,
                    Some(NodeParams::ComponentInlineParams(ComponentInlineParams {
                        selector: ComponentSelector::Expression(expr),
                    })),
                ),
                ComponentSelector::ExpressionList(expr_list) => Node::new(
                    KindT::Component,
                    Some(NodeParams::ComponentInlineParams(ComponentInlineParams {
                        selector: ComponentSelector::ExpressionList(expr_list),
                    })),
                ),
            };
            self.children
                .get_or_insert_with(Vec::new)
                .push(component_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }
    pub fn component_from_uri() {
        unimplemented!()
    }
    pub fn component_from_source() {
        unimplemented!()
    }
    pub fn label_from_uri() {
        unimplemented!()
    }
    pub fn label_from_source() {
        unimplemented!()
    }
    pub fn tooltip_from_uri() {
        unimplemented!()
    }
    pub fn tooltip_from_source() {
        unimplemented!()
    }
    pub fn transform(&mut self, params: TransformParams) -> Option<&mut Node> {
        if self.kind == KindT::Structure {
            let transform_node =
                Node::new(KindT::Transform, Some(NodeParams::TransformParams(params)));
            self.children
                .get_or_insert_with(Vec::new)
                .push(transform_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }
    pub fn _is_rotation_matrix() {
        unimplemented!()
    }
    // Component methods ------------------------------------------------------

    /// Add a representation for this component.
    /// :param type: the type of representation, defaults to 'cartoon'
    /// :return: a builder that handles operations at representation level

    pub fn representation(
        &mut self,
        representation_type: RepresentationTypeT,
    ) -> Option<&mut Node> {
        if self.kind == KindT::Component {
            let representation_node = Node::new(
                KindT::Representation,
                Some(NodeParams::RepresentationParams(RepresentationParams {
                    representation_type,
                })),
            );
            self.children
                .get_or_insert_with(Vec::new)
                .push(representation_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }

    pub fn label(&mut self, label: String) -> Option<&mut Node> {
        if self.kind == KindT::Component {
            let label_node = Node::new(
                KindT::Label,
                Some(NodeParams::LabelInlineParams(LabelInlineParams {
                    text: label,
                })),
            );
            self.children.get_or_insert_with(Vec::new).push(label_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }
    pub fn tooltip() {
        unimplemented!()
    }
    pub fn focus() {
        unimplemented!()
    }

    // Representation methods ------------------------------------------------------

    pub fn color_from_source() {
        unimplemented!()
    }
    pub fn color_from_uri() {
        unimplemented!()
    }
    // parent Kine => kindt:representation
    // node: kindt => kindt:color
    pub fn color(&mut self, color: ColorT, selector: ComponentSelector) -> Option<&mut Node> {
        if self.kind == KindT::Representation {
            let color_node = Node::new(
                KindT::Color,
                Some(NodeParams::ColorInlineParams(ColorInlineParams {
                    base: ComponentInlineParams { selector },
                    color,
                })),
            );
            self.children.get_or_insert_with(Vec::new).push(color_node);
            self.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }

    // GenericVisuals methods ------------------------------------------------------
    pub fn sphere() {
        unimplemented!()
    }
    pub fn line() {
        unimplemented!()
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum DescriptionFormatT {
    Markdown,
    Plaintext,
}

/// Metadata
///
/// The molviewspec metadata. High level info unrelated to
/// structure visualization.
///
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Metadata {
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description_format: Option<DescriptionFormatT>,
    pub timestamp: String,
}

/// This is the base MolViewSpec object containing
/// the root node and the metadata
///
/// Holds methods that modify the root node.
///
#[derive(Serialize, Deserialize, Debug)]
pub struct State {
    pub root: Node,
    pub metadata: Metadata,
}
impl State {
    pub fn new() -> Self {
        State {
            root: Node::new(KindT::Root, None),
            metadata: Metadata {
                version: "1".to_string(), // todo: update this
                timestamp: Utc::now().to_rfc3339().to_string(),
                ..Default::default()
            },
        }
    }
    /// Set Camera Location
    pub fn camera(&mut self, params: CameraParams) -> Option<&mut Node> {
        if self.root.kind == KindT::Root {
            let camera_node = Node::new(KindT::Camera, Some(NodeParams::CameraParams(params)));
            self.root
                .children
                .get_or_insert_with(Vec::new)
                .push(camera_node);
            self.root.children.as_mut().unwrap().last_mut()
        } else {
            None
        }
    }
    /// Set Canves Information Location
    pub fn canvas() {
        unimplemented!()
    }
    // Download a file
    pub fn download(&mut self, url: &str) -> Option<&mut Node> {
        self.root.download(url)
    }
    /// General Lines and Spheres
    pub fn generic_visuals() {
        unimplemented!()
    }
    pub fn to_url(&self) -> String {
        let json = serde_json::to_string(&self).expect("Json conversion");
        let encoded = urlencoding::encode(&json);
        let url = format!(
            "https://molstar.org/viewer/?mvs-format=mvsj&mvs-data={}",
            encoded
        );
        url
    }
}

/// Types of coumpounds: for pse I am only using PDB
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ParseFormatT {
    Mmcif,
    Bcif,
    Pdb,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename = "parse")]
pub struct ParseParams {
    pub format: ParseFormatT,
}

/// StructureType. Useful for specifying more complicated sets of structures
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum StructureTypeT {
    Model,
    Assembly,
    Symmetry,
    SymmetryMates,
}
impl Default for StructureTypeT {
    fn default() -> Self {
        StructureTypeT::Assembly
    }
}

/// Structure Params
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct StructureParams {
    #[serde(rename = "type")]
    pub structure_type: StructureTypeT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assembly_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assembly_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ijk_min: Option<(i32, i32, i32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ijk_max: Option<(i32, i32, i32)>,
}

/// Component Selector Type
///
/// Useful for specifying broad groups like 'all',
/// 'protein', etc.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ComponentSelectorT {
    All,
    Polymer,
    Protein,
    Nucleic,
    Branched,
    Ligand,
    Ion,
    Water,
}
/// Component Expresssion
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ComponentExpression {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_entity_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_asym_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_asym_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_seq_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_seq_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pdbx_pdb_ins_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beg_label_seq_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_label_seq_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beg_auth_seq_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_auth_seq_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residue_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_atom_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_atom_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_symbol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atom_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atom_index: Option<i32>,
}

/// Representation Type
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationTypeT {
    BallAndStick,
    Cartoon,
    Surface,
}

/// Color Names
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ColorNamesT {
    Aliceblue,
    Antiquewhite,
    Aqua,
    Aquamarine,
    Azure,
    Beige,
    Bisque,
    Black,
    Blanchedalmond,
    Blue,
    Blueviolet,
    Brown,
    Burlywood,
    Cadetblue,
    Chartreuse,
    Chocolate,
    Coral,
    Cornflowerblue,
    Cornsilk,
    Crimson,
    Cyan,
    Darkblue,
    Darkcyan,
    Darkgoldenrod,
    Darkgray,
    Darkgreen,
    Darkgrey,
    Darkkhaki,
    Darkmagenta,
    Darkolivegreen,
    Darkorange,
    Darkorchid,
    Darkred,
    Darksalmon,
    Darkseagreen,
    Darkslateblue,
    Darkslategray,
    Darkslategrey,
    Darkturquoise,
    Darkviolet,
    Deeppink,
    Deepskyblue,
    Dimgray,
    Dimgrey,
    Dodgerblue,
    Firebrick,
    Floralwhite,
    Forestgreen,
    Fuchsia,
    Gainsboro,
    Ghostwhite,
    Gold,
    Goldenrod,
    Gray,
    Green,
    Greenyellow,
    Grey,
    Honeydew,
    Hotpink,
    Indianred,
    Indigo,
    Ivory,
    Khaki,
    Lavender,
    Lavenderblush,
    Lawngreen,
    Lemonchiffon,
    Lightblue,
    Lightcoral,
    Lightcyan,
    Lightgoldenrodyellow,
    Lightgray,
    Lightgreen,
    Lightgrey,
    Lightpink,
    Lightsalmon,
    Lightseagreen,
    Lightskyblue,
    Lightslategray,
    Lightslategrey,
    Lightsteelblue,
    Lightyellow,
    Lime,
    Limegreen,
    Linen,
    Magenta,
    Maroon,
    Mediumaquamarine,
    Mediumblue,
    Mediumorchid,
    Mediumpurple,
    Mediumseagreen,
    Mediumslateblue,
    Mediumspringgreen,
    Mediumturquoise,
    Mediumvioletred,
    Midnightblue,
    Mintcream,
    Mistyrose,
    Moccasin,
    Navajowhite,
    Navy,
    Oldlace,
    Olive,
    Olivedrab,
    Orange,
    Orangered,
    Orchid,
    Palegoldenrod,
    Palegreen,
    Paleturquoise,
    Palevioletred,
    Papayawhip,
    Peachpuff,
    Peru,
    Pink,
    Plum,
    Powderblue,
    Purple,
    Red,
    Rosybrown,
    Royalblue,
    Saddlebrown,
    Salmon,
    Sandybrown,
    Seagreen,
    Seashell,
    Sienna,
    Silver,
    Skyblue,
    Slateblue,
    Slategray,
    Slategrey,
    Snow,
    Springgreen,
    Steelblue,
    Tan,
    Teal,
    Thistle,
    Tomato,
    Turquoise,
    Violet,
    Wheat,
    White,
    Whitesmoke,
    Yellow,
    Yellowgreen,
}

/// Color Type: Named or Hex
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ColorT {
    Named(ColorNamesT),
    Hex(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RepresentationParams {
    #[serde(rename = "type")]
    pub representation_type: RepresentationTypeT,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum SchemaT {
    WholeStructure,
    Entity,
    Chain,
    AuthChain,
    Residue,
    AuthResidue,
    ResidueRange,
    AuthResidueRange,
    Atom,
    AuthAtom,
    AllAtomic,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SchemaFormatT {
    Cif,
    Bcif,
    Json,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataFromUriParams {
    pub uri: String,
    pub format: SchemaFormatT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_index: Option<i32>,
    #[serde(rename = "schema")]
    pub schema_: SchemaT,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataFromSourceParams {
    pub category_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_index: Option<i32>,
    #[serde(rename = "schema")]
    pub schema_: SchemaT,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ComponentInlineParams {
    pub selector: ComponentSelector,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ComponentSelector {
    Selector(ComponentSelectorT),
    Expression(ComponentExpression),
    ExpressionList(Vec<ComponentExpression>),
}
impl Default for ComponentSelector {
    fn default() -> Self {
        ComponentSelector::Selector(ComponentSelectorT::All)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ComponentFromUriParams {
    #[serde(flatten)]
    pub base: DataFromUriParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_values: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ComponentFromSourceParams {
    #[serde(flatten)]
    pub base: DataFromSourceParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_values: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColorInlineParams {
    #[serde(flatten)]
    pub base: ComponentInlineParams,
    pub color: ColorT,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColorFromUriParams {
    #[serde(flatten)]
    pub base: DataFromUriParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColorFromSourceParams {
    #[serde(flatten)]
    pub base: DataFromSourceParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelInlineParams {
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelFromUriParams {
    #[serde(flatten)]
    pub base: DataFromUriParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelFromSourceParams {
    #[serde(flatten)]
    pub base: DataFromSourceParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TooltipInlineParams {
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TooltipFromUriParams {
    #[serde(flatten)]
    pub base: DataFromUriParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TooltipFromSourceParams {
    #[serde(flatten)]
    pub base: DataFromSourceParams,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FocusInlineParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<(f64, f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub up: Option<(f64, f64, f64)>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TransformParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translation: Option<(f64, f64, f64)>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SphereParams {
    pub position: (f64, f64, f64),
    pub radius: f64,
    pub color: ColorT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tooltip: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LineParams {
    pub position1: (f64, f64, f64),
    pub position2: (f64, f64, f64),
    pub radius: f64,
    pub color: ColorT,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tooltip: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct CameraParams {
    pub target: (f64, f64, f64),
    pub position: (f64, f64, f64),
    #[serde(skip_serializing_if = "Option::is_none")]
    pub up: Option<(f64, f64, f64)>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CanvasParams {
    pub background_color: ColorT,
}

#[derive(Debug, Serialize, Deserialize, Validate, Clone)]
pub struct DownloadParams {
    pub url: String,
}
