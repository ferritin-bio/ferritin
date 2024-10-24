// This Rust code provides:

// 1. Structs representing the components of your API spec.
// 2. Serialization and deserialization using serde.
// 3. A builder pattern for the `LineParams` struct (you can implement similar builders for other complex structs if needed).
// 4. Validation logic using the `validator` crate, including custom validation for the `RepresentationParams` type.
// 5. A function to generate the OpenAPI spec from these Rust structures.

// Note that this is a basic implementation and might need some adjustments:

// - The `generate_openapi_spec` function creates a basic OpenAPI structure. You might need to add more details to fully represent your spec.
// - The validation for array lengths and enum values is implemented, but you might want to add more specific validations for other fields.
// - The `Node` struct uses a generic `serde_json::Value` for its `params` field. You might want to create a more specific enum or struct to represent the different types of params based on the node kind.
// - You might want to implement custom serialization/deserialization for some fields, especially for color names and enums.
use openapi::{Components, Info, OpenApi, Paths};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use validator::{Validate, ValidationError};

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

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct Metadata {
    pub version: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub description_format: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Node {
    pub kind: String,
    pub params: Option<serde_json::Value>,
    pub children: Option<Vec<Node>>,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct ParseParams {
    pub format: String,
}

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct RepresentationParams {
    #[validate(custom = "validate_representation_type")]
    pub r#type: String,
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

fn validate_representation_type(r#type: &str) -> Result<(), ValidationError> {
    match r#type {
        "ball_and_stick" | "cartoon" | "surface" => Ok(()),
        _ => Err(ValidationError::new("Invalid representation type")),
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

impl LineParams {
    pub fn builder() -> LineParamsBuilder {
        LineParamsBuilder::default()
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
                    "LineParams".to_string(),
                    serde_json::to_value(LineParams::default()).unwrap(),
                );
                schemas.insert(
                    "Metadata".to_string(),
                    serde_json::to_value(Metadata::default()).unwrap(),
                );
                schemas.insert(
                    "Node".to_string(),
                    serde_json::to_value(Node::default()).unwrap(),
                );
                schemas.insert(
                    "ParseParams".to_string(),
                    serde_json::to_value(ParseParams::default()).unwrap(),
                );
                schemas.insert(
                    "RepresentationParams".to_string(),
                    serde_json::to_value(RepresentationParams::default()).unwrap(),
                );
                schemas.insert(
                    "SphereParams".to_string(),
                    serde_json::to_value(SphereParams::default()).unwrap(),
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
    let line_params = LineParams::builder()
        .position1(vec![0.0, 0.0, 0.0])
        .position2(vec![1.0, 1.0, 1.0])
        .radius(0.5)
        .color("red".to_string())
        .build()
        .unwrap();

    println!("Line params: {:?}", line_params);

    let openapi_spec = generate_openapi_spec();
    println!(
        "OpenAPI spec: {}",
        serde_json::to_string_pretty(&openapi_spec).unwrap()
    );
}
