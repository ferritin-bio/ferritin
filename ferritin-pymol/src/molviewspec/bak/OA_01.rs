use openapi::{Components, Info, OpenApi, Paths};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::{Validate, ValidationError};

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

// Implement builder patterns for complex objects
impl CameraParams {
    pub fn builder() -> CameraParamsBuilder {
        CameraParamsBuilder::default()
    }
}

#[derive(Default)]
pub struct CameraParamsBuilder {
    target: Option<Vec<f64>>,
    position: Option<Vec<f64>>,
    up: Option<Vec<f64>>,
}

impl CameraParamsBuilder {
    pub fn target(mut self, target: Vec<f64>) -> Self {
        self.target = Some(target);
        self
    }

    pub fn position(mut self, position: Vec<f64>) -> Self {
        self.position = Some(position);
        self
    }

    pub fn up(mut self, up: Vec<f64>) -> Self {
        self.up = Some(up);
        self
    }

    pub fn build(self) -> Result<CameraParams, String> {
        let target = self.target.ok_or("target is required")?;
        let position = self.position.ok_or("position is required")?;
        let up = self.up.ok_or("up is required")?;

        let params = CameraParams {
            target,
            position,
            up,
        };
        params
            .validate()
            .map_err(|e| format!("Validation error: {:?}", e))?;
        Ok(params)
    }
}

// Function to generate OpenAPI spec
pub fn generate_openapi_spec() -> OpenApi {
    OpenApi {
        openapi: "3.0.0".to_string(),
        info: Info {
            title: "MolViewSpec Node Schema OpenAPI".to_string(),
            version: "0.1".to_string(),
            ..Default::default()
        },
        paths: Paths::new(),
        components: Some(Components {
            schemas: {
                let mut schemas = HashMap::new();
                schemas.insert(
                    "CameraParams".to_string(),
                    serde_json::to_value(CameraParams::default()).unwrap(),
                );
                schemas.insert(
                    "CanvasParams".to_string(),
                    serde_json::to_value(CanvasParams::default()).unwrap(),
                );
                schemas.insert(
                    "ColorFromSourceParams".to_string(),
                    serde_json::to_value(ColorFromSourceParams::default()).unwrap(),
                );
                schemas.insert(
                    "ColorFromUriParams".to_string(),
                    serde_json::to_value(ColorFromUriParams::default()).unwrap(),
                );
                schemas.insert(
                    "ColorInlineParams".to_string(),
                    serde_json::to_value(ColorInlineParams::default()).unwrap(),
                );
                schemas
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}
