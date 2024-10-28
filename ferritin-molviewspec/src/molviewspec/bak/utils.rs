use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// This is a simplified version of BaseModel for Rust
pub trait BaseModel: Serialize + for<'de> Deserialize<'de> {
    fn fields() -> Vec<&'static str>;
}

pub fn make_params<T: BaseModel>(
    values: Option<&HashMap<String, serde_json::Value>>,
    more_values: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut result = HashMap::new();
    let values = values.unwrap_or(&HashMap::new());

    for field in T::fields() {
        if let Some(value) = more_values.get(field) {
            result.insert(field.to_string(), value.clone());
        } else if let Some(value) = values.get(field) {
            result.insert(field.to_string(), value.clone());
        }
        // Note: We're not handling default values here as it's not straightforward in Rust
        // You might need to implement this differently based on your specific needs
    }

    result
}

pub fn get_major_version_tag() -> String {
    let version_parts: Vec<&str> = env!("CARGO_PKG_VERSION").split('.').collect();
    let major = version_parts.get(0).unwrap_or(&"0");
    let minor = version_parts.get(1).unwrap_or(&"0");

    if major == &"0" {
        format!("{}.{}", major, minor)
    } else {
        major.to_string()
    }
}
