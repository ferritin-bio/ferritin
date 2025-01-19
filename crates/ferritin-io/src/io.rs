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
    len += read_line(reader, &mut buf)?;
    len += consume_hashtag_line(reader)?;

    let src = reader.fill_buf()?;
    if let Some(buf) = src.get(..5) {
        if buf == LOOP_DENOTE {
            println!("Loop here!");
            // len += process_loop();
        } else if buf[0] == b'_' {
            println!("Map of K/V here");
            // len += process_kv();
        }
    }

    println!("Length: {}", len);
    println!("Buffer: {:?}", &buf);

    // let mut len = match read_definition(reader, record.definition_mut()) {
    //     Ok(0) => return Ok(0),
    //     Ok(n) => n,
    //     Err(e) => return Err(e),
    // };

    // len += read_line(reader, record.sequence_mut())?;
    // len += consume_plus_line(reader)?;
    // len += read_line(reader, record.quality_scores_mut())?;

    Ok(len)
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
        read_cif_record(&mut reader, &mut cif);

        Ok(())
    }
}
