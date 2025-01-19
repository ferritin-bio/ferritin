use anyhow::{Error, Result};
use std::collections::HashMap;
use std::io::{self, BufRead, Read};

//  Cif Definition --------------------------------------------------------------------------------

struct CIF {
    file_data: HashMap<String, String>,
    tables: Option<Vec<Table>>,
}
impl CIF {
    pub fn new(name: String) -> Result<Self> {
        let mut file_data = HashMap::new();
        file_data.insert("entry".to_string(), name);
        Ok(CIF {
            file_data,
            tables: None,
        })
    }
}

#[derive(Clone)]
enum Value {
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    // Add other types as needed
}

struct Table {
    name: String,
    num_columns: usize,
    data: HashMap<String, Vec<Value>>,
}
impl Table {}

//  IO Fns  --------------------------------------------------------------------------------

const BLOCK_DELIMITER: u8 = b'#';
const LINE_FEED: u8 = b'\n';
const CARRIAGE_RETURN: u8 = b'\r';

// pub(super) fn read_record<R>(reader: &mut R, record: &mut Record) -> io::Result<usize>
// where
//     R: BufRead,
// {
//     record.clear();

//     let mut len = match read_definition(reader, record.definition_mut()) {
//         Ok(0) => return Ok(0),
//         Ok(n) => n,
//         Err(e) => return Err(e),
//     };

//     len += read_line(reader, record.sequence_mut())?;
//     len += consume_plus_line(reader)?;
//     len += read_line(reader, record.quality_scores_mut())?;

//     Ok(len)
// }

pub(super) fn read_cif_record<R>(reader: &mut R, cif: &mut CIF) -> io::Result<usize>
where
    R: BufRead,
{
    // 1. read the first line. start the dict.
    // 2. process each block
    //    1. if pure k/b add to the top-level dict.
    //        1. 2 values for line
    //        2. value spread over multiple lines
    //    2. if `loop_` then its a table
    //
    //
    let LOOP_DENOTE = b"loop_";
    let mut len = 0;
    let mut buf: Vec<u8> = Vec::new();
    let mut data: HashMap<String, String> = HashMap::new();
    let mut table_data: HashMap<String, Vec<String>> = HashMap::new();

    len += read_line(reader, &mut buf)?;
    len += consume_hashtag_line(reader)?;
    println!("Length: {}", len);

    // len += read_line(reader, record.sequence_mut())?;
    // len += consume_plus_line(reader)?;
    // len += read_line(reader, record.quality_scores_mut())?;
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }
        // consume the DataBlocks.
        // data blocks are either K/V pairs or tables.
        // tables are denoted by `loop_`
        if let Some(buf) = buf.get(..5) {
            if buf == LOOP_DENOTE {
                println!("Loop here!");
                len += process_table(reader, &mut table_data)?;
                println!("PT: {:?}", len);
                // len += consume_hashtag_line(reader)?;
            } else if buf[0] == b'_' {
                println!("Map of K/V here");
                len += process_kv(reader, &mut data)?;
                println!("PKV: {:?}", len);
                // len += consume_hashtag_line(reader)?;
            } else if buf[0] == b'#' {
                len += consume_hashtag_line(reader)?;
                println!("#: {:?}", len)
            } else {
                let break_string = String::from_utf8_lossy(&buf).trim().to_string();
                println!("Break Buffer: {:?}", break_string);
            }
        }
    }

    // println!("Data Keys: {:?}", data.keys());
    // println!("Table Data Keys: {:?}", table_data.keys());
    Ok(len)
}

fn process_table<R>(reader: &mut R, hashmap: &mut HashMap<String, Vec<String>>) -> io::Result<usize>
where
    R: BufRead,
{
    let mut buf = Vec::new();
    let mut total_len = 0;

    // first line is loop_
    let len = read_line(reader, &mut buf)?;
    total_len += len;
    assert_eq!(buf, b"loop_");

    // i now want to iterate through each line
    // lines that start with `_` are field lines
    // they defin the header
    // after that are lines with data.
    // how should i iterate through the lines?

    // Collect headers
    let mut headers = Vec::new();
    buf.clear();

    // Read headers (lines starting with '_')
    loop {
        if let Ok(len) = read_line(reader, &mut buf) {
            if buf.is_empty() || buf[0] != b'_' {
                break;
            }
            headers.push(String::from_utf8_lossy(&buf).trim().to_string());
            total_len += len;
            buf.clear();
        }
    }

    loop {
        if buf.is_empty() || buf[0] == b'#' {
            break;
        }

        // Process the data line
        let line = String::from_utf8_lossy(&buf).trim().to_string();

        // Split the line by whitespace and process values
        let values: Vec<&str> = line.split_whitespace().collect();

        // Match values with headers
        for (header, value) in headers.iter().zip(values.iter()) {
            if let Some(vec) = hashmap.get_mut(header) {
                vec.push(value.to_string());
            }
        }

        // Read next line
        buf.clear();
        match read_line(reader, &mut buf) {
            Ok(len) => total_len += len,
            Err(_) => break,
        }
    }

    Ok(total_len)
}

fn process_kv<R>(reader: &mut R, hashmap: &mut HashMap<String, String>) -> io::Result<usize>
where
    R: BufRead,
{
    let mut buf = Vec::new();
    let mut total_len = 0;

    // Read first line containing the key
    let len = read_line(reader, &mut buf)?;
    total_len += len;

    if let Ok(line) = String::from_utf8(buf.clone()) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if !parts.is_empty() {
            let key = parts[0].trim().to_string();
            let mut value = String::new();

            // If we have both key and value on the same line
            if parts.len() >= 2 {
                value = parts[1..].join(" ").trim().to_string();
            } else {
                // Read next line to check for semicolon-delimited text
                buf.clear();
                let next_len = read_line(reader, &mut buf)?;
                total_len += next_len;

                if let Ok(next_line) = String::from_utf8(buf.clone()) {
                    if next_line.trim_start().starts_with(';') {
                        // Handle semicolon-delimited text
                        let mut content = String::new();

                        // Continue reading until we find an ending semicolon
                        loop {
                            buf.clear();
                            let line_len = read_line(reader, &mut buf)?;
                            if line_len == 0 {
                                break;
                            } // EOF
                            total_len += line_len;

                            if let Ok(line) = String::from_utf8(buf.clone()) {
                                if line.trim_start().starts_with(';') {
                                    break;
                                }
                                content.push_str(&line);
                            }
                        }
                        value = content.trim().to_string();
                    } else {
                        value = next_line.trim().to_string();
                    }
                }
            }

            hashmap.insert(key, value);
        }
    }

    Ok(total_len)
}

fn read_line<R>(reader: &mut R, buf: &mut Vec<u8>) -> io::Result<usize>
where
    R: BufRead,
{
    match reader.read_until(LINE_FEED, buf) {
        Ok(0) => Ok(0),
        Ok(n) => {
            if buf.ends_with(&[LINE_FEED]) {
                buf.pop();
                if buf.ends_with(&[CARRIAGE_RETURN]) {
                    buf.pop();
                }
            }
            Ok(n)
        }
        Err(e) => Err(e),
    }
}

fn consume_line<R>(reader: &mut R) -> io::Result<usize>
where
    R: BufRead,
{
    use memchr::memchr;
    let mut is_eol = false;
    let mut len = 0;
    loop {
        let src = reader.fill_buf()?;
        if src.is_empty() || is_eol {
            break;
        }
        let n = match memchr(LINE_FEED, src) {
            Some(i) => {
                is_eol = true;
                i + 1
            }
            None => src.len(),
        };
        reader.consume(n);
        len += n;
    }
    Ok(len)
}

fn read_u8<R>(reader: &mut R) -> io::Result<u8>
where
    R: Read,
{
    let mut buf = [0; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn consume_plus_line<R>(reader: &mut R) -> io::Result<usize>
where
    R: BufRead,
{
    const PREFIX: u8 = b'+';

    match read_u8(reader)? {
        PREFIX => consume_line(reader).map(|n| n + 1),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid description prefix",
        )),
    }
}

fn consume_hashtag_line<R>(reader: &mut R) -> io::Result<usize>
where
    R: BufRead,
{
    const PREFIX: u8 = b'#';

    match read_u8(reader)? {
        PREFIX => consume_line(reader).map(|n| n + 1),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid description prefix",
        )),
    }
}

//  CIF  Definition --------------------------------------------------------------------------------

/// A Cif reader.
pub struct Reader<R> {
    inner: R,
}

impl<R> Reader<R> {
    /// Returns a reference to the underlying reader.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }
    /// Returns a mutable reference to the underlying reader.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }
    /// Unwraps and returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R> Reader<R>
where
    R: BufRead,
{
    /// Creates a CIF reader.
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    // /// Reads a CIF record.
    // pub fn read_record(&mut self, record: &mut Record) -> io::Result<usize> {
    //     read_record(&mut self.inner, record)
    // }

    // /// Returns an iterator over records starting from the current stream position.
    // pub fn records(&mut self) -> Records<'_, R> {
    //     Records::new(self)
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use ferritin_test_data::TestFile;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_basic_read() -> Result<()> {
        // basic buffer use
        let mut buf = Vec::new();
        let data = b"noodles\n";
        let mut reader = &data[..];
        buf.clear();
        let out = read_line(&mut reader, &mut buf)?;
        println!("{:?}", out);
        assert_eq!(buf, b"noodles");

        // Test Cif Header
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        println!("{:?}", prot_file);
        let f = File::open(prot_file)?;
        let mut reader = BufReader::new(f);
        let mut buf = Vec::new();
        let out = read_line(&mut reader, &mut buf)?;
        println!("{:?}", out);
        assert_eq!(buf, b"data_101M");
        Ok(())
    }
    #[test]
    fn test_cif_read() -> Result<()> {
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        println!("{:?}", prot_file);
        let f = File::open(prot_file)?;
        let mut reader = BufReader::new(f);
        let mut cif = CIF::new("Test".to_string())?;
        let cif_len = read_cif_record(&mut reader, &mut cif)?;
        assert_eq!(cif_len, 198551);
        Ok(())
    }
}
