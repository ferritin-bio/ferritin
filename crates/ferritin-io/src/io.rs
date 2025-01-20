use polars::prelude::CsvReadOptions;
use polars::prelude::*;
use std::collections::HashMap;
use std::io::{self, BufRead, Read};

//  Cif Definition --------------------------------------------------------------------------------

// Represent a CIF structure
struct CifFile {
    name: String,
    data_blocks: HashMap<String, CifBlock>,
}

enum BlockType {
    SINGLE_VALUE,
    MULTIPLE_VALUE,
}

// Simple key-value pairs
// Loop data as dataframes
struct CifBlock {
    block_type: BlockType,
    data: DataFrame,
}

impl CifFile {
    fn new(name: String) -> Self {
        CifFile {
            name: name,
            data_blocks: HashMap::new(),
        }
    }
    fn add_block(&mut self, name: String, block: CifBlock) {
        self.data_blocks.insert(name, block);
    }
}

impl CifBlock {
    fn new(block_type: BlockType, data: DataFrame) -> Self {
        CifBlock {
            block_type: block_type,
            data: data,
        }
    }
}

//  IO Fns  --------------------------------------------------------------------------------

const BLOCK_DELIMITER: u8 = b'#';
const LINE_FEED: u8 = b'\n';
const CARRIAGE_RETURN: u8 = b'\r';
const LOOP_DENOTE: &[u8; 5] = b"loop_";

pub(super) fn read_cif_record<R>(reader: &mut R, ciffile: &mut CifFile) -> io::Result<usize>
where
    R: BufRead,
{
    // 1. read the first line. start the dict.
    // 2. process each block
    //    1. if pure k/b add to the top-level dict.
    //        1. 2 values for line
    //        2. value spread over multiple lines
    //    2. if `loop_` then its a table

    let mut len = 0;
    let mut buf: Vec<u8> = Vec::new();
    len += read_line(reader, &mut buf)?;

    // todo: implement
    let file_code = String::from_utf8(buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    len += consume_hashtag_line(reader)?;

    // consume the DataBlocks.
    // data blocks are either K/V pairs or tables.
    // tables are denoted by `loop_`
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            break;
        }
        if let Some(buf) = buf.get(..5) {
            if buf == LOOP_DENOTE {
                len += process_table(reader, ciffile)?;
            } else if buf[0] == b'_' {
                len += process_kv_block(reader, ciffile)?;
            } else if buf[0] == b'#' {
                len += consume_hashtag_line(reader)?;
            }
        }
    }
    Ok(len)
}

fn process_table<R>(reader: &mut R, cif: &mut CifFile) -> io::Result<usize>
where
    R: BufRead,
{
    let mut buf = Vec::new();
    let mut total_len = 0;

    // first line is loop_
    let len = read_line(reader, &mut buf)?;
    total_len += len;
    assert_eq!(buf, b"loop_");

    // then the headers
    let mut headers = Vec::new();
    buf.clear();
    loop {
        // Peek at the next line
        if !reader.fill_buf()?.starts_with(b"_") {
            break;
        }
        let len = read_line(reader, &mut buf)?;
        headers.push(String::from_utf8_lossy(&buf).trim().to_string());
        total_len += len;
        buf.clear();
    }
    println!("Headers: {:?}", headers);

    // then the data
    let mut collected = Vec::new();
    loop {
        buf.clear();
        let len = read_line(reader, &mut buf)?;
        if len == 0 {
            break; // EOF
        }
        let line = String::from_utf8_lossy(&buf).trim().to_string();
        if line == "#" {
            break;
        }
        collected.push(line);
    }

    let parse_opts = CsvParseOptions::default()
        .with_separator(b' ')
        .with_truncate_ragged_lines(true)
        .with_try_parse_dates(true)
        .with_quote_char(Some(b'\''));

    fn collapse_whitespace(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    // let table_text = &collected.join("\n");
    let table_text = collected
        .iter()
        .map(|line| collapse_whitespace(line))
        .collect::<Vec<_>>()
        .join("\n");

    println!("Table Text: \n {:?}", table_text);

    // let header_names = Arc::from(
    //     headers
    //         .iter()
    //         .map(|h| h.trim_start_matches('_').into())
    //         .collect::<Box<[PlSmallStr]>>(), // Collect directly into Box<[T]>
    // );

    // println!("Header Names: \n {:?}", header_names);

    let df = CsvReadOptions::default()
        .with_has_header(false)
        // .with_columns(Some(header_names))
        .with_parse_options(parse_opts)
        .into_reader_with_file_handle(std::io::Cursor::new(table_text))
        .finish()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    println!("DF: {:?}", &df);

    let table_name = headers[0].clone();
    let block = CifBlock::new(BlockType::MULTIPLE_VALUE, df);
    cif.add_block(table_name.to_string(), block);
    Ok(total_len)
}

fn process_kv_block<R>(reader: &mut R, cif: &mut CifFile) -> io::Result<usize>
where
    R: BufRead,
{
    let mut total_len = 0;
    let mut hashmap = HashMap::new();

    // loop through the KV block and add k/v pairs to the hashmap
    loop {
        // Peek at the next line without consuming it
        let mut buf: Vec<u8> = Vec::new();
        let len = reader.fill_buf()?.len();
        if len == 0 {
            break;
        }
        if reader.fill_buf()?.starts_with(b"#") {
            break;
        }
        let kv_len = process_kv(reader, &mut hashmap)?;
        total_len += kv_len;
    }

    let series: Vec<Series> = hashmap
        .into_iter()
        .map(|(name, data)| {
            Series::from_any_values(name.as_str().into(), &[data.as_str().into()], false)
                .expect("Failed to create series")
        })
        .collect();

    let table_name = series[0].name().to_string();
    let df = DataFrame::from_iter(series);
    let block = CifBlock::new(BlockType::SINGLE_VALUE, df);
    cif.add_block(table_name.to_string(), block);
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
    assert_ne!(buf, b"loop_");

    total_len += len;

    if let Ok(line) = String::from_utf8(buf.clone()) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if !parts.is_empty() {
            let key = parts[0].trim().to_string();
            let mut value = String::new();

            // If we have both key and value on the same line
            if parts.len() == 2 {
                value = parts[1].trim().to_string();
            } else if parts.len() >= 2 {
                panic!("There should not be multiple parts ....{:?}", parts)
            } else {
                // Read next line to check for semicolon-delimited text
                buf.clear();
                let next_len = read_line(reader, &mut buf)?;
                total_len += next_len;

                if let Ok(next_line) = String::from_utf8(buf.clone()) {
                    if next_line.trim_start().starts_with(';') {
                        // Handle semicolon-delimited text
                        let mut content = String::new();
                        loop {
                            buf.clear();
                            let line_len = read_line(reader, &mut buf)?;
                            if line_len == 0 {
                                break;
                            }
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
        ch => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid hashtag line prefix: {}", ch),
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

        let mut cif = CifFile::new("101M".to_string()); //
        let cif_len = read_cif_record(&mut reader, &mut cif)?;
        // assert_eq!(cif_len, 176763);

        assert_eq!(cif.name, "101M");
        // hmm. sometimes 39 and sometimes 42? whaaat?
        // assert_eq!(cif.data_blocks.len(), 42);

        println!("Data Block Kets: {:?}", &cif.data_blocks.keys());
        let table_01 = &cif.data_blocks.get("_entry.id").unwrap().data;
        println!("{:?}", table_01);
        assert_eq!(table_01.shape(), (2, 4));
        Ok(())
    }
}
