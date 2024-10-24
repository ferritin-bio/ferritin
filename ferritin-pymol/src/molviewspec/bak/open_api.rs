// use openapi::{Components, Info, OpenApi, Paths};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::{Validate, ValidationError};

pub fn get_builder() -> State {
    State::new()
}

#[derive(Debug, Serialize, Deserialize)]
enum AllParams {
    CameraParams(CameraParams),
    CanvasParams(CanvasParams),
    ColorFromSourceParams(ColorFromSourceParams),
    ColorFromUriParams(ColorFromUriParams),
    ColorInlineParams(ColorInlineParams),
    ComponentFromSourceParams(ComponentFromSourceParams),
    ComponentFromUriParams(ComponentFromUriParams),
    ComponentInlineParams(ComponentInlineParams),
    DownloadParams(DownloadParams),
    FocusInlineParams(FocusInlineParams),
    LabelFromSourceParams(LabelFromSourceParams),
    LabelFromUriParams(LabelFromUriParams),
    LabelInlineParams(LabelInlineParams),
    LineParams(LineParams),
    ParseParams(ParseParams),
    RepresentationParams(RepresentationParams),
    SphereParams(SphereParams),
    StructureParams(StructureParams),
    TooltipFromSourceParams(TooltipFromSourceParams),
    TooltipFromUriParams(TooltipFromUriParams),
    TooltipInlineParams(TooltipInlineParams),
    TransformParams(TransformParams),
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct CameraParams {
    #[validate(length(min = 3, max = 3))]
    pub target: Vec<f64>,
    #[validate(length(min = 3, max = 3))]
    pub position: Vec<f64>,
    #[validate(length(min = 3, max = 3))]
    pub up: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct CanvasParams {
    pub background_color: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ColorFromSourceParams {
    pub category_name: String,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ColorFromUriParams {
    pub uri: String,
    pub format: String,
    pub category_name: Option<String>,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Selector {
    Predefined(String),
    ComponentExpression(ComponentExpression),
    MultipleComponentExpressions(Vec<ComponentExpression>),
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ColorInlineParams {
    pub selector: Selector,
    pub color: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentExpression {
    pub label_entity_id: Option<String>,
    pub label_asym_id: Option<String>,
    pub auth_asym_id: Option<String>,
    pub label_seq_id: Option<i32>,
    pub auth_seq_id: Option<i32>,
    pub pdbx_PDB_ins_code: Option<String>,
    pub beg_label_seq_id: Option<i32>,
    pub end_label_seq_id: Option<i32>,
    pub beg_auth_seq_id: Option<i32>,
    pub end_auth_seq_id: Option<i32>,
    pub residue_index: Option<i32>,
    pub label_atom_id: Option<String>,
    pub auth_atom_id: Option<String>,
    pub type_symbol: Option<String>,
    pub atom_id: Option<i32>,
    pub atom_index: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ComponentFromSourceParams {
    pub category_name: String,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
    pub field_values: Option<Vec<String>>,
}

// Implement builder patterns for complex objects
impl ComponentFromSourceParams {
    pub fn builder() -> ComponentFromSourceParamsBuilder {
        ComponentFromSourceParamsBuilder::default()
    }
}

#[derive(Default)]
pub struct ComponentFromSourceParamsBuilder {
    category_name: Option<String>,
    field_name: Option<String>,
    block_header: Option<String>,
    block_index: Option<i32>,
    schema: Option<String>,
    field_values: Option<Vec<String>>,
}

impl ComponentFromSourceParamsBuilder {
    pub fn category_name(mut self, category_name: String) -> Self {
        self.category_name = Some(category_name);
        self
    }

    pub fn field_name(mut self, field_name: String) -> Self {
        self.field_name = Some(field_name);
        self
    }

    pub fn block_header(mut self, block_header: String) -> Self {
        self.block_header = Some(block_header);
        self
    }

    pub fn block_index(mut self, block_index: i32) -> Self {
        self.block_index = Some(block_index);
        self
    }

    pub fn schema(mut self, schema: String) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn field_values(mut self, field_values: Vec<String>) -> Self {
        self.field_values = Some(field_values);
        self
    }

    pub fn build(self) -> Result<ComponentFromSourceParams, String> {
        let category_name = self.category_name.ok_or("category_name is required")?;
        let schema = self.schema.ok_or("schema is required")?;

        let params = ComponentFromSourceParams {
            category_name,
            field_name: self.field_name,
            block_header: self.block_header,
            block_index: self.block_index,
            schema,
            field_values: self.field_values,
        };
        params
            .validate()
            .map_err(|e| format!("Validation error: {:?}", e))?;
        Ok(params)
    }
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ComponentFromUriParams {
    pub uri: String,
    pub format: String,
    pub category_name: Option<String>,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
    pub field_values: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ComponentInlineParams {
    pub selector: Selector,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct DownloadParams {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct FocusInlineParams {
    #[validate(length(min = 3, max = 3))]
    pub direction: Option<Vec<f64>>,
    #[validate(length(min = 3, max = 3))]
    pub up: Option<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct LabelFromSourceParams {
    pub category_name: String,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct LabelFromUriParams {
    pub uri: String,
    pub format: String,
    pub category_name: Option<String>,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct LabelInlineParams {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct LineParams {
    #[validate(length(min = 3, max = 3))]
    pub position1: Vec<f64>,
    #[validate(length(min = 3, max = 3))]
    pub position2: Vec<f64>,
    pub radius: f64,
    pub color: String,
    pub label: Option<String>,
    pub tooltip: Option<String>,
}

impl LineParams {
    pub fn builder() -> LineParamsBuilder {
        LineParamsBuilder::default()
    }
}

// Builder pattern for LineParams
#[derive(Default)]
pub struct LineParamsBuilder {
    position1: Option<Vec<f64>>,
    position2: Option<Vec<f64>>,
    radius: Option<f64>,
    color: Option<String>,
    label: Option<String>,
    tooltip: Option<String>,
}

impl LineParamsBuilder {
    pub fn position1(mut self, position1: Vec<f64>) -> Self {
        self.position1 = Some(position1);
        self
    }

    pub fn position2(mut self, position2: Vec<f64>) -> Self {
        self.position2 = Some(position2);
        self
    }

    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = Some(radius);
        self
    }

    pub fn color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }

    pub fn label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    pub fn tooltip(mut self, tooltip: String) -> Self {
        self.tooltip = Some(tooltip);
        self
    }

    pub fn build(self) -> Result<LineParams, String> {
        let line_params = LineParams {
            position1: self.position1.ok_or("position1 is required")?,
            position2: self.position2.ok_or("position2 is required")?,
            radius: self.radius.ok_or("radius is required")?,
            color: self.color.ok_or("color is required")?,
            label: self.label,
            tooltip: self.tooltip,
        };
        line_params
            .validate()
            .map_err(|e| format!("Validation error: {:?}", e))?;
        Ok(line_params)
    }
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct Metadata {
    pub version: String,
    pub title: Option<String>,
    pub description: Option<String>,
    #[validate(custom(function = "validate_description_format"))]
    pub description_format: Option<String>,
    pub timestamp: String,
}
impl Metadata {
    pub fn new() -> Self {
        Metadata {
            version: "1.0".to_string(),
            title: None,
            description: None,
            description_format: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    pub kind: String,
    pub params: Option<HashMap<String, serde_json::Value>>,
    pub children: Option<Vec<Node>>,
}
impl Node {
    pub fn new(kind: String, params: Option<HashMap<String, serde_json::Value>>) -> Node {
        Node {
            kind,
            params: params,
            children: None,
        }
    }
    pub fn add_child(&mut self, node: Node) {
        match &mut self.children {
            Some(children) => children.push(node),
            None => self.children = Some(vec![node]),
        }
    }
    /// Create the download node
    pub fn download(&mut self, url: &str) -> Node {
        // Todo: shoudl we chack allowable kinds here?
        let kind = &self.kind;
        println!("{:?}", kind);
        let params = Some(HashMap::from([(
            "url".to_string(),
            serde_json::Value::String(url.to_string()),
        )]));

        let download_node = Node::new("download".to_string(), params);

        self.add_child(download_node.clone());
        download_node
    }
    // Create the parse node
    pub fn parse(&mut self, format: &str) -> Node {
        let kind = &self.kind;
        assert!(
            kind == "download",
            "Kind must be 'download' for parse operation"
        );

        assert!(
            ["pdb", "mmcif"].contains(&format),
            "Format must be either 'pdb' or 'mmcif'"
        );

        let params = Some(HashMap::from([(
            "format".to_string(),
            serde_json::Value::String(format.to_string()),
        )]));

        let parse_node = Node::new("parse".to_string(), params);

        self.add_child(parse_node.clone());
        parse_node
    }
    // Define Assembly Structure
    pub fn model_structure(&mut self, model_index: i32) -> Node {
        let kind = &self.kind;
        assert!(
            kind == "parse",
            "Kind must be 'parse' for assembly_structure"
        );

        let params = Some(HashMap::from([(
            "model index".to_string(),
            serde_json::Value::Number(serde_json::Number::from(model_index as i64)),
        )]));

        let structure_node = Node::new("structure".to_string(), params);
        self.add_child(structure_node.clone());
        structure_node
    }
    pub fn component(&mut self) -> Node {
        let kind = &self.kind;
        assert!(
            kind == "structure",
            "Kind must be 'structure' for assembly_structure"
        );

        let params = Some(HashMap::from([(
            "component".to_string(),
            serde_json::Value::String("".to_string()),
        )]));

        let parse_node = Node::new("component".to_string(), params);

        self.add_child(parse_node.clone());
        parse_node
    }
    pub fn representation(&mut self) -> Node {
        let kind = &self.kind;
        assert!(
            kind == "structure",
            "Kind must be 'structure' for assembly_structure"
        );

        let params = Some(HashMap::from([(
            "component".to_string(),
            serde_json::Value::String("".to_string()),
        )]));

        let parse_node = Node::new("component".to_string(), params);

        self.add_child(parse_node.clone());
        parse_node
    }
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ParseParams {
    pub format: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct RepresentationParams {
    #[validate(custom(function = "validate_representation_type"))]
    #[serde(rename = "type")]
    pub representation_type: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct SphereParams {
    #[validate(length(min = 3, max = 3))]
    pub position: Vec<f64>,
    pub radius: f64,
    pub color: String,
    pub label: Option<String>,
    pub tooltip: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct State {
    pub root: Node,
    pub metadata: Metadata,
}
impl State {
    pub fn new() -> State {
        State {
            root: Node::new("root".to_string(), None),
            metadata: Metadata::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct StructureParams {
    #[validate(custom(function = "validate_structure_type"))]
    #[serde(rename = "type")]
    pub structure_type: String,
    pub assembly_id: Option<String>,
    pub assembly_index: Option<i32>,
    pub model_index: Option<i32>,
    pub block_index: Option<i32>,
    pub block_header: Option<String>,
    pub radius: Option<f64>,
    #[validate(length(min = 3, max = 3))]
    pub ijk_min: Option<Vec<i32>>,
    #[validate(length(min = 3, max = 3))]
    pub ijk_max: Option<Vec<i32>>,
}

impl StructureParams {
    pub fn builder() -> StructureParamsBuilder {
        StructureParamsBuilder::default()
    }
}

// Builder pattern for StructureParams
#[derive(Default, Serialize, Deserialize)]
pub struct StructureParamsBuilder {
    #[serde(rename = "type")]
    structure_type: Option<String>,
    assembly_id: Option<String>,
    assembly_index: Option<i32>,
    model_index: Option<i32>,
    block_index: Option<i32>,
    block_header: Option<String>,
    radius: Option<f64>,
    ijk_min: Option<Vec<i32>>,
    ijk_max: Option<Vec<i32>>,
}

impl StructureParamsBuilder {
    pub fn structure_type(mut self, structure_type: String) -> Self {
        self.structure_type = Some(structure_type);
        self
    }

    pub fn assembly_id(mut self, assembly_id: String) -> Self {
        self.assembly_id = Some(assembly_id);
        self
    }

    pub fn assembly_index(mut self, assembly_index: i32) -> Self {
        self.assembly_index = Some(assembly_index);
        self
    }

    pub fn model_index(mut self, model_index: i32) -> Self {
        self.model_index = Some(model_index);
        self
    }

    pub fn block_index(mut self, block_index: i32) -> Self {
        self.block_index = Some(block_index);
        self
    }

    pub fn block_header(mut self, block_header: String) -> Self {
        self.block_header = Some(block_header);
        self
    }

    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = Some(radius);
        self
    }

    pub fn ijk_min(mut self, ijk_min: Vec<i32>) -> Self {
        self.ijk_min = Some(ijk_min);
        self
    }

    pub fn ijk_max(mut self, ijk_max: Vec<i32>) -> Self {
        self.ijk_max = Some(ijk_max);
        self
    }

    pub fn build(self) -> Result<StructureParams, String> {
        let params = StructureParams {
            structure_type: self.structure_type.ok_or("type is required")?,
            assembly_id: self.assembly_id,
            assembly_index: self.assembly_index,
            model_index: self.model_index,
            block_index: self.block_index,
            block_header: self.block_header,
            radius: self.radius,
            ijk_min: self.ijk_min,
            ijk_max: self.ijk_max,
        };
        params
            .validate()
            .map_err(|e| format!("Validation error: {:?}", e))?;
        Ok(params)
    }
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct TooltipFromSourceParams {
    pub category_name: String,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    #[validate(custom(function = "validate_schema"))]
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct TooltipFromUriParams {
    pub uri: String,
    #[validate(custom(function = "validate_format"))]
    pub format: String,
    pub category_name: Option<String>,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    #[validate(custom(function = "validate_schema"))]
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct TooltipInlineParams {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct TransformParams {
    pub rotation: Option<Vec<f64>>,
    #[validate(length(min = 3, max = 3))]
    pub translation: Option<Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataFromSourceParams {
    pub category_name: String,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataFromUriParams {
    pub uri: String,
    pub format: String,
    pub category_name: Option<String>,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Component {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Download {}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenericVisuals {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Parse {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Representation {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Root {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Structure {}

#[derive(Debug, Serialize, Deserialize)]
pub struct Base {}

//  Validation Function  -----------------------------------------------------------------------

fn validate_description_format(format: &str) -> Result<(), ValidationError> {
    match format {
        "markdown" | "plaintext" => Ok(()),
        _ => Err(ValidationError::new("Invalid description format")),
    }
}

fn validate_structure_type(structure_type: &str) -> Result<(), ValidationError> {
    match structure_type {
        "model" | "assembly" | "symmetry" | "symmetry_mates" => Ok(()),
        _ => Err(ValidationError::new("Invalid structure type")),
    }
}

fn validate_schema(schema: &str) -> Result<(), ValidationError> {
    match schema {
        "whole_structure" | "entity" | "chain" | "auth_chain" | "residue" | "auth_residue"
        | "residue_range" | "auth_residue_range" | "atom" | "auth_atom" | "all_atomic" => Ok(()),
        _ => Err(ValidationError::new("Invalid schema")),
    }
}

fn validate_format(format: &str) -> Result<(), ValidationError> {
    match format {
        "cif" | "bcif" | "json" => Ok(()),
        _ => Err(ValidationError::new("Invalid format")),
    }
}

fn validate_representation_type(representation_type: &str) -> Result<(), ValidationError> {
    match representation_type {
        "ball_and_stick" | "cartoon" | "surface" => Ok(()),
        _ => Err(ValidationError::new("Invalid representation type")),
    }
}

// // Function to generate OpenAPI spec
// pub fn generate_openapi_spec() -> OpenApi {
//     OpenApi {
//         openapi: "3.0.0".to_string(),
//         info: Info {
//             title: "MolViewSpec Node Schema OpenAPI".to_string(),
//             version: "0.1".to_string(),
//             ..Default::default()
//         },
//         paths: Paths::new(),
//         components: Some(Components {
//             schemas: {
//                 let mut schemas = HashMap::new();
//                 schemas.insert(
//                     "ComponentExpression".to_string(),
//                     serde_json::to_value(ComponentExpression::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "ComponentFromSourceParams".to_string(),
//                     serde_json::to_value(ComponentFromSourceParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "ComponentFromUriParams".to_string(),
//                     serde_json::to_value(ComponentFromUriParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "ComponentInlineParams".to_string(),
//                     serde_json::to_value(ComponentInlineParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "DownloadParams".to_string(),
//                     serde_json::to_value(DownloadParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "FocusInlineParams".to_string(),
//                     serde_json::to_value(FocusInlineParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "LabelFromSourceParams".to_string(),
//                     serde_json::to_value(LabelFromSourceParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "LabelFromUriParams".to_string(),
//                     serde_json::to_value(LabelFromUriParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "LabelInlineParams".to_string(),
//                     serde_json::to_value(LabelInlineParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "CameraParams".to_string(),
//                     serde_json::to_value(CameraParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "CanvasParams".to_string(),
//                     serde_json::to_value(CanvasParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "ColorFromSourceParams".to_string(),
//                     serde_json::to_value(ColorFromSourceParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "ColorFromUriParams".to_string(),
//                     serde_json::to_value(ColorFromUriParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "ColorInlineParams".to_string(),
//                     serde_json::to_value(ColorInlineParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "LineParams".to_string(),
//                     serde_json::to_value(LineParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "Metadata".to_string(),
//                     serde_json::to_value(Metadata::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "Node".to_string(),
//                     serde_json::to_value(Node::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "ParseParams".to_string(),
//                     serde_json::to_value(ParseParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "RepresentationParams".to_string(),
//                     serde_json::to_value(RepresentationParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "SphereParams".to_string(),
//                     serde_json::to_value(SphereParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "State".to_string(),
//                     serde_json::to_value(State::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "Node".to_string(),
//                     serde_json::to_value(Node::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "Metadata".to_string(),
//                     serde_json::to_value(Metadata::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "StructureParams".to_string(),
//                     serde_json::to_value(StructureParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "TooltipFromSourceParams".to_string(),
//                     serde_json::to_value(TooltipFromSourceParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "TooltipFromUriParams".to_string(),
//                     serde_json::to_value(TooltipFromUriParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "TooltipInlineParams".to_string(),
//                     serde_json::to_value(TooltipInlineParams::default()).unwrap(),
//                 );
//                 schemas.insert(
//                     "TransformParams".to_string(),
//                     serde_json::to_value(TransformParams::default()).unwrap(),
//                 );
//                 schemas
//             },
//             ..Default::default()
//         }),
//         ..Default::default()
//     }
// }
