use ferritin_cellscape::cellscape::StructureFlatten;
use ferritin_core::load_model;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let model = load_model("data/101m.cif")?;

    let doc = model.flatten_structure();
    svg::save("simple_02.svg", &doc)?;
    println!("SVG has been created as 'simple_02.svg'");
    Ok(())
}
