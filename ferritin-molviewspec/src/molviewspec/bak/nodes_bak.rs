use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Node {
    pub kind: KindT,
    #[serde(skip_serializing_if = "Option::is_none")]
    // pub params: Option<HashMap<String, serde_json::Value>>,
    pub params: Option<AllParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<Node>>,
}
impl Node {
    // Common to All Nodes
    pub fn new(kind: KindT, params: Option<AllParams>) -> Node {
        Node {
            kind: kind,
            ..Default::default()
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
    // Only Applicable to the Download Node
    pub fn add_download_info(&mut self, url: Dow) -> Option<&Node> {
        let params = Some(HashMap::from([(
            "url".to_string(),
            serde_json::Value::String(url),
        )]));

        match self.kind {
            KindT::Download => {
                self.params = params;
                Some(self)
            }
            _ => None,
        }
    }
    // //
    // pub fn create_parser(&mut self, parseformat: String) -> Option<&Node> {
    //     let params = Some(HashMap::from([(
    //         "format".to_string(),
    //         serde_json::Value::String(parseformat),
    //     )]));

    //     let mut parse = Node::new(KindT::Parse);
    //     parse.params = params;

    //     match self.kind {
    //         KindT::Download => {
    //             self.params = params;
    //             Some(self)
    //         }
    //         _ => None,
    //     }
    // }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum DescriptionFormatT {
    Markdown,
    Plaintext,
}

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

#[derive(Serialize, Deserialize, Debug)]
pub struct State {
    pub root: Node,
    pub metadata: Metadata,
}
impl State {
    pub fn new() -> Self {
        State {
            root: Node::new(KindT::Root),
            metadata: Metadata {
                version: "1.0".to_string(), // todo: update this
                timestamp: Utc::now().to_rfc3339().to_string(),
                ..Default::default()
            },
        }
    }
    pub fn download(&mut self, url: String) -> Option<&Node> {
        let mut download = Node::new(KindT::Download);
        download.add_download_info(url);
        self.root.add_child(download);
        Some(
            self.root
                .children
                .as_ref()
                .expect("Children vector should exist")
                .last()
                .expect("At least one child should exist after adding"),
        )
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum ParseFormatT {
    Mmcif,
    Bcif,
    Pdb,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ParseParams {
    format: ParseFormatT,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum StructureTypeT {
    Model,
    Assembly,
    Symmetry,
    SymmetryMates,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StructureParams {
    #[serde(rename = "type")]
    structure_type: StructureTypeT,
    #[serde(skip_serializing_if = "Option::is_none")]
    assembly_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    assembly_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    radius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ijk_min: Option<(i32, i32, i32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ijk_max: Option<(i32, i32, i32)>,
}

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationTypeT {
    BallAndStick,
    Cartoon,
    Surface,
}

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ColorT {
    Named(ColorNamesT),
    Hex(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RepresentationParams {
    #[serde(rename = "type")]
    representation_type: RepresentationTypeT,
}

#[derive(Serialize, Deserialize, Debug)]
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

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum SchemaFormatT {
    Cif,
    Bcif,
    Json,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DataFromUriParams {
    uri: String,
    format: SchemaFormatT,
    #[serde(skip_serializing_if = "Option::is_none")]
    category_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_index: Option<i32>,
    #[serde(rename = "schema")]
    schema_: SchemaT,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DataFromSourceParams {
    category_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_index: Option<i32>,
    #[serde(rename = "schema")]
    schema_: SchemaT,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ComponentInlineParams {
    selector: ComponentSelector,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ComponentSelector {
    Selector(ComponentSelectorT),
    Expression(ComponentExpression),
    ExpressionList(Vec<ComponentExpression>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ComponentFromUriParams {
    #[serde(flatten)]
    base: DataFromUriParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_values: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ComponentFromSourceParams {
    #[serde(flatten)]
    base: DataFromSourceParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    field_values: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ColorInlineParams {
    #[serde(flatten)]
    base: ComponentInlineParams,
    color: ColorT,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ColorFromUriParams {
    #[serde(flatten)]
    base: DataFromUriParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ColorFromSourceParams {
    #[serde(flatten)]
    base: DataFromSourceParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LabelInlineParams {
    text: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LabelFromUriParams {
    #[serde(flatten)]
    base: DataFromUriParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LabelFromSourceParams {
    #[serde(flatten)]
    base: DataFromSourceParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TooltipInlineParams {
    text: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TooltipFromUriParams {
    #[serde(flatten)]
    base: DataFromUriParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TooltipFromSourceParams {
    #[serde(flatten)]
    base: DataFromSourceParams,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FocusInlineParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    direction: Option<(f32, f32, f32)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    up: Option<(f32, f32, f32)>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransformParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    rotation: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    translation: Option<(f32, f32, f32)>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CameraParams {
    target: (f32, f32, f32),
    position: (f32, f32, f32),
    up: (f32, f32, f32),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CanvasParams {
    background_color: ColorT,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SphereParams {
    position: (f32, f32, f32),
    radius: f32,
    color: ColorT,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tooltip: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LineParams {
    position1: (f32, f32, f32),
    position2: (f32, f32, f32),
    radius: f32,
    color: ColorT,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tooltip: Option<String>,
}
