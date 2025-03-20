use ferritin_cellscape::cellscape::StructureFlatten;
use ferritin_core::load_structure;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let ac = load_structure("data/101m.cif").unwrap();

    let doc = ac.flatten_structure();
    svg::save("simple_02.svg", &doc)?;
    println!("SVG has been created as 'simple_02.svg'");
    Ok(())
}
