use std::io::BufRead;

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

//  CIO  Definition --------------------------------------------------------------------------------

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

    /// Reads a CIF record.
    ///
    pub fn read_record(&mut self, record: &mut Record) -> io::Result<usize> {
        read_record(&mut self.inner, record)
    }

    /// Returns an iterator over records starting from the current stream position.
    pub fn records(&mut self) -> Records<'_, R> {
        Records::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Definition;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_basic_read() {
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(prot_file).unwrap();
        let ac = AtomCollection::from(&pdb);

        let input = b"_cell.length_a  10.5";
        let mut reader = Reader::new(&input[..]);
        let mut record = Record::default();
        assert_eq!(reader.read_record(&mut record).unwrap(), 1);
    }
}
