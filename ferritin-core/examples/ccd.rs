// //! BinaryCIF implementation following the official specification
// //! Based on https://github.com/dsehnal/BinaryCIF
// use serde::{Deserialize, Serialize};
// use std::fs::File;
// use std::io::BufReader;
// use std::io::Read;
// use std::path::Path;

// pub const VERSION: &str = "0.3.0";

// #[derive(Debug, Serialize, Deserialize)]
// pub struct BinaryCifFile {
//     #[serde(rename = "dataBlocks")]
//     pub data_blocks: Vec<DataBlock>,
//     // Make these optional since they might not be in the file
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub version: Option<String>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub encoder: Option<String>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct DataBlock {
//     pub categories: Vec<Category>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub header: Option<String>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct Category {
//     #[serde(rename = "rowCount")]
//     pub row_count: u32,
//     pub columns: Vec<Column>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub name: Option<String>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct Column {
//     pub name: String,
//     pub data: EncodedData,
//     /// The mask represents presence/absence of values:
//     /// 0 = Value is present
//     /// 1 = . = value not specified
//     /// 2 = ? = value unknown
//     pub mask: Option<EncodedData>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// pub struct EncodedData {
//     pub encoding: Vec<Encoding>,
//     #[serde(with = "serde_bytes")]
//     pub data: Vec<u8>,
// }

// #[derive(Debug, Serialize, Deserialize)]
// #[serde(tag = "kind")]
// pub enum Encoding {
//     #[serde(rename = "ByteArray")]
//     ByteArray {
//         #[serde(rename = "type")]
//         data_type: DataType,
//     },
//     #[serde(rename = "FixedPoint")]
//     FixedPoint {
//         factor: f64,
//         #[serde(rename = "srcType")]
//         src_type: FloatDataType,
//     },
//     #[serde(rename = "RunLength")]
//     RunLength {
//         #[serde(rename = "srcType")]
//         src_type: IntDataType,
//         #[serde(rename = "srcSize")]
//         src_size: u32,
//     },
//     #[serde(rename = "Delta")]
//     Delta {
//         origin: i32,
//         #[serde(rename = "srcType")]
//         src_type: IntDataType,
//     },
//     #[serde(rename = "IntervalQuantization")]
//     IntervalQuantization {
//         min: f64,
//         max: f64,
//         #[serde(rename = "numSteps")]
//         num_steps: u32,
//         #[serde(rename = "srcType")]
//         src_type: FloatDataType,
//     },
//     #[serde(rename = "IntegerPacking")]
//     IntegerPacking {
//         #[serde(rename = "byteCount")]
//         byte_count: u32,
//         #[serde(rename = "isUnsigned")]
//         is_unsigned: bool,
//         #[serde(rename = "srcSize")]
//         src_size: u32,
//     },
//     #[serde(rename = "StringArray")]
//     StringArray {
//         #[serde(rename = "dataEncoding")]
//         data_encoding: Vec<Encoding>,
//         #[serde(rename = "stringData")]
//         string_data: String,
//         #[serde(rename = "offsetEncoding")]
//         offset_encoding: Vec<Encoding>,
//         offsets: Vec<u8>,
//     },
// }

// /// Integer data types used in encodings
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[repr(u8)]
// pub enum IntDataType {
//     Int8 = 1,
//     Int16 = 2,
//     Int32 = 3,
//     Uint8 = 4,
//     Uint16 = 5,
//     Uint32 = 6,
// }

// /// Floating point data types used in encodings
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[repr(u8)]
// pub enum FloatDataType {
//     Float32 = 32,
//     Float64 = 33,
// }

// /// Combined data type enum that can be either integer or float
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(untagged)]
// pub enum DataType {
//     Int(IntDataType),
//     Float(FloatDataType),
// }

// /// CIF value types representing presence/absence of data
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum CifValueStatus {
//     Present = 0,      // Value exists
//     NotSpecified = 1, // "." in CIF
//     Unknown = 2,      // "?" in CIF
// }

// /// Types of encoding methods available
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum EncodingType {
//     ByteArray,
//     FixedPoint,
//     RunLength,
//     Delta,
//     IntervalQuantization,
//     IntegerPacking,
//     StringArray,
// }

// impl IntDataType {
//     /// Get the size in bytes for this integer type
//     pub fn size_in_bytes(&self) -> usize {
//         match self {
//             IntDataType::Int8 | IntDataType::Uint8 => 1,
//             IntDataType::Int16 | IntDataType::Uint16 => 2,
//             IntDataType::Int32 | IntDataType::Uint32 => 4,
//         }
//     }

//     /// Check if this is an unsigned type
//     pub fn is_unsigned(&self) -> bool {
//         match self {
//             IntDataType::Uint8 | IntDataType::Uint16 | IntDataType::Uint32 => true,
//             _ => false,
//         }
//     }
// }

// impl FloatDataType {
//     /// Get the size in bytes for this float type
//     pub fn size_in_bytes(&self) -> usize {
//         match self {
//             FloatDataType::Float32 => 4,
//             FloatDataType::Float64 => 8,
//         }
//     }
// }

// impl DataType {
//     /// Get the size in bytes for this data type
//     pub fn size_in_bytes(&self) -> usize {
//         match self {
//             DataType::Int(i) => i.size_in_bytes(),
//             DataType::Float(f) => f.size_in_bytes(),
//         }
//     }

//     /// Check if this is an integer type
//     pub fn is_integer(&self) -> bool {
//         matches!(self, DataType::Int(_))
//     }

//     /// Check if this is a float type
//     pub fn is_float(&self) -> bool {
//         matches!(self, DataType::Float(_))
//     }
// }

// /// Errors that can occur during encoding/decoding
// #[derive(Debug, thiserror::Error)]
// pub enum BinaryCifError {
//     #[error("Invalid data type")]
//     InvalidDataType,
//     #[error("Invalid encoding")]
//     InvalidEncoding,
//     #[error("Invalid mask value")]
//     InvalidMask,
//     #[error("IO error: {0}")]
//     Io(#[from] std::io::Error),
//     #[error("MessagePack error: {0}")]
//     MessagePack(#[from] rmp_serde::decode::Error),
// }

// type BinaryCifResult<T> = std::result::Result<T, BinaryCifError>; // Changed from Result<T>

// impl BinaryCifFile {
//     pub fn from_reader<R: Read>(reader: R) -> BinaryCifResult<Self> {
//         let file: BinaryCifFile = rmp_serde::from_read(reader)?;
//         Ok(file)
//     }

//     pub fn from_bytes(bytes: &[u8]) -> BinaryCifResult<Self> {
//         let file: BinaryCifFile = rmp_serde::from_slice(bytes)?;
//         Ok(file)
//     }
// }
// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // Load and parse file
//     // let bytes = std::fs::read("data/ccd/components.bcif")?;
//     let bytes = std::fs::read("/Users/zcpowers/Documents/Projects/ferritin/data/biotite-1.0.1/src/biotite/structure/info/ccd/components.bcif")?;
//     let bcif = BinaryCifFile::from_bytes(&bytes)?;
//     // Print structure overview
//     println!("File version: {:?}", bcif.version);
//     println!("Number of data blocks: {:?}", bcif.data_blocks.len());

//     for block in &bcif.data_blocks {
//         println!("\nBlock: {:?}", block.header);
//         for category in &block.categories {
//             println!(
//                 "  Category {:?} ({:?} rows, {:?} columns)",
//                 category.name,
//                 category.row_count,
//                 category.columns.len()
//             );
//         }
//     }

//     Ok(())
// }
// //
//
//
fn main() {}
