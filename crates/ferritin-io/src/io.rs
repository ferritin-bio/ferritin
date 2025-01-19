use anyhow::{Error, Result};
use std::io::{self, BufRead, Read};

//  Cif Definition --------------------------------------------------------------------------------
struct File {
    version: String,
    encoder: String,
    data_blocks: Vec<DataBlock>,
}

struct DataBlock {
    header: String,
    categories: Vec<Category>,
}

struct Category {
    name: String,
    row_count: usize,
    columns: Vec<Column>,
}

struct Column {
    name: String,
    data: Data,
    mask: Option<Data>,
}

struct Data {
    data: Vec<u8>,
    encoding: Vec<String>,
}

//  IO Fns  --------------------------------------------------------------------------------

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

fn read_line<R>(reader: &mut R, buf: &mut Vec<u8>) -> Result<usize>
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
        Err(e) => Err(e.into()),
    }
}

fn consume_line<R>(reader: &mut R) -> Result<usize>
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

// fn read_u8<R>(reader: &mut R) -> Result<u8>
// where
//     R: Read,
// {
//     let mut buf = [0; 1];
//     reader.read_exact(&mut buf)?;
//     Ok(buf[0])
// }

// fn consume_plus_line<R>(reader: &mut R) -> Result<usize>
// where
//     R: BufRead,
// {
//     const PREFIX: u8 = b'+';

//     match read_u8(reader)? {
//         PREFIX => consume_line(reader).map(|n| n + 1),
//         _ => Err(Error::new(
//             ErrorKind::InvalidData,
//             "invalid description prefix",
//         )),
//     }
// }

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
    // use crate::record::Definition;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_basic_read() -> Result<()> {
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        println!("{:?}", prot_file);

        let mut buf = Vec::new();
        let data = b"noodles\n";
        let mut reader = &data[..];
        buf.clear();

        let out = read_line(&mut reader, &mut buf)?;
        println!("{:?}", out);
        // let reader = Reader::new();

        // let (pdb, _) = pdbtbx::open(prot_file).unwrap();
        // let ac = AtomCollection::from(&pdb);

        // let input = b"_cell.length_a  10.5";
        // let mut reader = Reader::new(&input[..]);
        // let mut record = Record::default();
        // assert_eq!(reader.read_record(&mut record).unwrap(), 1);
        Ok(())
    }
}
