use openapi::{Components, Info, OpenApi, Paths};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::{Validate, ValidationError};

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Selector {
    Predefined(String),
    ComponentExpression(ComponentExpression),
    MultipleComponentExpressions(Vec<ComponentExpression>),
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
                    "ComponentExpression".to_string(),
                    serde_json::to_value(ComponentExpression::default()).unwrap(),
                );
                schemas.insert(
                    "ComponentFromSourceParams".to_string(),
                    serde_json::to_value(ComponentFromSourceParams::default()).unwrap(),
                );
                schemas.insert(
                    "ComponentFromUriParams".to_string(),
                    serde_json::to_value(ComponentFromUriParams::default()).unwrap(),
                );
                schemas.insert(
                    "ComponentInlineParams".to_string(),
                    serde_json::to_value(ComponentInlineParams::default()).unwrap(),
                );
                schemas.insert(
                    "DownloadParams".to_string(),
                    serde_json::to_value(DownloadParams::default()).unwrap(),
                );
                schemas.insert(
                    "FocusInlineParams".to_string(),
                    serde_json::to_value(FocusInlineParams::default()).unwrap(),
                );
                schemas.insert(
                    "LabelFromSourceParams".to_string(),
                    serde_json::to_value(LabelFromSourceParams::default()).unwrap(),
                );
                schemas.insert(
                    "LabelFromUriParams".to_string(),
                    serde_json::to_value(LabelFromUriParams::default()).unwrap(),
                );
                schemas.insert(
                    "LabelInlineParams".to_string(),
                    serde_json::to_value(LabelInlineParams::default()).unwrap(),
                );
                schemas
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

// Example usage
fn main() {
    let component_params = ComponentFromSourceParams::builder()
        .category_name("my_category".to_string())
        .schema("whole_structure".to_string())
        .build()
        .unwrap();

    println!("Component params: {:?}", component_params);

    let openapi_spec = generate_openapi_spec();
    println!(
        "OpenAPI spec: {}",
        serde_json::to_string_pretty(&openapi_spec).unwrap()
    );
}
