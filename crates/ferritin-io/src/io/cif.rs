#[derive(Debug, Serialize, Deserialize)]
struct File {
    version: String,
    encoder: String,
    #[serde(rename = "dataBlocks")]
    data_blocks: Vec<DataBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataBlock {
    header: String,
    categories: Vec<Category>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Category {
    name: String,
    #[serde(rename = "rowCount")]
    row_count: usize,
    columns: Vec<Column>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Column {
    name: String,
    data: Data,
    mask: Option<Data>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Data {
    data: Vec<u8>,
    encoding: Vec<Encoding>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum Encoding {
    ByteArray {
        kind: String,
        #[serde(rename = "type")]
        type_: ByteArrayEnum,
    },
    FixedPoint {
        kind: String,
        factor: i32,
        #[serde(rename = "srcType")]
        src_type: FixedPointEnum,
    },
    RunLength {
        kind: String,
        #[serde(rename = "srcType")]
        src_type: i32,
        #[serde(rename = "srcSize")]
        src_size: i32,
    },
    Delta {
        kind: String,
        origin: f64,
        #[serde(rename = "srcType")]
        src_type: i32,
    },
    IntegerPacking {
        kind: String,
        #[serde(rename = "byteCount")]
        byte_count: i32,
        #[serde(rename = "srcSize")]
        src_size: i32,
        #[serde(rename = "isUnsigned")]
        is_unsigned: bool,
    },
    StringArray {
        kind: String,
        #[serde(rename = "dataEncoding")]
        data_encoding: Box<Vec<Encoding>>,
        #[serde(rename = "stringData")]
        string_data: String,
        #[serde(rename = "offsetEncoding")]
        offset_encoding: Box<Vec<Encoding>>,
        #[serde(with = "serde_bytes")]
        offsets: Vec<u8>,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[repr(u32)]
#[serde(into = "i32", from = "i32")]
enum ByteArrayEnum {
    Int8 = 1,
    Int16 = 2,
    Int32 = 3,
    Uint8 = 4,
    Uint16 = 5,
    Uint32 = 6,
    Float32 = 32,
    Float64 = 33,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[repr(u32)]
#[serde(into = "i32", from = "i32")]
enum FixedPointEnum {
    Float32 = 32,
    Float64 = 33,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
}
