// This code provides:

// 1. Rust structs representing the components of your API spec.
// 2. Serialization and deserialization using serde.
// 3. Builder patterns for complex objects (e.g., `Node`, `StructureParams`).
// 4. Basic validation logic for some structs (e.g., `StructureParams`, `TooltipFromUriParams`).
// 5. A function to generate the OpenAPI spec from Rust structures.

// To use this code, you'll need to add the following dependencies to your `Cargo.toml`:

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct State {
    pub root: Node,
    pub metadata: Metadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Node {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<Node>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub version: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub description_format: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StructureParams {
    #[serde(rename = "type")]
    pub structure_type: String,
    pub assembly_id: Option<String>,
    pub assembly_index: Option<i32>,
    pub model_index: Option<i32>,
    pub block_index: Option<i32>,
    pub block_header: Option<String>,
    pub radius: Option<f64>,
    pub ijk_min: Option<[i32; 3]>,
    pub ijk_max: Option<[i32; 3]>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TooltipFromSourceParams {
    pub category_name: String,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TooltipFromUriParams {
    pub uri: String,
    pub format: String,
    pub category_name: Option<String>,
    pub field_name: Option<String>,
    pub block_header: Option<String>,
    pub block_index: Option<i32>,
    pub schema: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TooltipInlineParams {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TransformParams {
    pub rotation: Option<Vec<f64>>,
    pub translation: Option<[f64; 3]>,
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

// Builder patterns
impl Node {
    pub fn new(kind: String) -> Self {
        Node {
            kind,
            params: None,
            children: None,
        }
    }

    pub fn with_params(mut self, params: HashMap<String, serde_json::Value>) -> Self {
        self.params = Some(params);
        self
    }

    pub fn with_children(mut self, children: Vec<Node>) -> Self {
        self.children = Some(children);
        self
    }
}

impl StructureParams {
    pub fn new(structure_type: String) -> Self {
        StructureParams {
            structure_type,
            assembly_id: None,
            assembly_index: None,
            model_index: None,
            block_index: None,
            block_header: None,
            radius: None,
            ijk_min: None,
            ijk_max: None,
        }
    }

    // Add builder methods for optional fields...
}

// Validation logic
impl StructureParams {
    pub fn validate(&self) -> Result<(), String> {
        if !["model", "assembly", "symmetry", "symmetry_mates"]
            .contains(&self.structure_type.as_str())
        {
            return Err("Invalid structure type".to_string());
        }
        if let Some(ijk_min) = self.ijk_min {
            if ijk_min.len() != 3 {
                return Err("ijk_min must have exactly 3 elements".to_string());
            }
        }
        if let Some(ijk_max) = self.ijk_max {
            if ijk_max.len() != 3 {
                return Err("ijk_max must have exactly 3 elements".to_string());
            }
        }
        Ok(())
    }
}

impl TooltipFromUriParams {
    pub fn validate(&self) -> Result<(), String> {
        if !["cif", "bcif", "json"].contains(&self.format.as_str()) {
            return Err("Invalid format".to_string());
        }
        // Add more validation as needed
        Ok(())
    }
}

// OpenAPI spec generation
pub fn generate_openapi_spec() -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.0.0",
        "info": {
            "title": "MolViewSpec Node Schema OpenAPI",
            "version": "0.1"
        },
        "paths": {},
        "components": {
            "schemas": {
                "State": {
                    "type": "object",
                    "properties": {
                        "root": { "$ref": "#/components/schemas/Node" },
                        "metadata": { "$ref": "#/components/schemas/Metadata" }
                    },
                    "required": ["root", "metadata"]
                },
                "Node": {
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string" },
                        "params": {
                            "type": "object",
                            "additionalProperties": true
                        },
                        "children": {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/Node" }
                        }
                    },
                    "required": ["kind"]
                },
                // Add other schema definitions...
            }
        }
    })
}

fn main() {
    // Example usage
    let node = Node::new("root".to_string())
        .with_params(HashMap::new())
        .with_children(vec![]);

    let state = State {
        root: node,
        metadata: Metadata {
            version: "0.1".to_string(),
            title: Some("Example".to_string()),
            description: None,
            description_format: None,
            timestamp: "2023-05-10T12:00:00Z".to_string(),
        },
    };

    let json = serde_json::to_string_pretty(&state).unwrap();
    println!("{}", json);

    let spec = generate_openapi_spec();
    println!("{}", serde_json::to_string_pretty(&spec).unwrap());
}
