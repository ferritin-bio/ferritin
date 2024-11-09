use ferritin_cellscape::cellscape::StructureFlatten;
use ferritin_core::AtomCollection;
use pdbtbx;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let (pdb, _) = pdbtbx::open("data/101m.cif").unwrap();
    let ac = AtomCollection::from(&pdb);
    let doc = ac.flatten_structure();
    svg::save("simple_02.svg", &doc)?;
    println!("SVG has been created as 'simple_02.svg'");
    Ok(())
}
