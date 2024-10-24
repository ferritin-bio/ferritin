use itertools::assert_equal;
use pseutils::pymolparsing::colors::Color;
use pseutils::pymolparsing::parsing::{CoordSet, CustomValue, SettingsEnum};
use pseutils::pymolparsing::representation::RepBitmask;
use pseutils::PSEData;
const TEST_OUTPUT_DIR: &str = "./test_temporary";

#[test]
fn test_load_pse_data_molecule_only() {
    //https://users.rust-lang.org/t/serde-untagged-enum-ruins-precise-errors/54128/2
    // https://www.gustavwengel.dk/serde-untagged-enum-errors-are-bad
    let deserialized: PSEData = PSEData::load("tests/data/example_molecule_only.pse").unwrap();
    // deserialized.to_json("tests/data/example_molecule_only.json");
    assert!(deserialized.version == 3000000);
    assert!(
        deserialized
            .get_setting(SettingsEnum::Orthoscopic)
            .unwrap()
            .value
            == CustomValue::Integer(0)
    );
    // println!("{:?}", deserialized.settings);
}

#[test]
fn test_load_pse_data_molecule_selection() {
    let deserialized: PSEData = PSEData::load("tests/data/example.pse").unwrap();
    // let _ = deserialized.to_json("tests/data/example.json");
    assert!(deserialized.version == 3000000);
}

#[test]
fn test_pdb_00() {
    let psedata: PSEData = PSEData::load("tests/data/example.pse").unwrap();
    // this has a Molecule and a selection
    let names = psedata.get_session_names();
    assert_eq!(names.len(), 2);

    let mols = psedata.get_molecule_data();
    assert_eq!(mols.len(), 1);

    // check coordinates. theres 1 mon and 1519 atoms
    let coord_sets: Vec<&CoordSet> = mols.iter().flat_map(|mol| mol.coord_set.iter()).collect();
    assert_eq!(coord_sets.len(), 1);

    let coords_01 = coord_sets[0].get_coords_as_vec();
    assert_eq!(coords_01.len(), 1519);

    let coords_01_atom_01 = coords_01;
    assert_equal(coords_01_atom_01[0], [50.873, 32.978, 2.387]);
    assert_equal(coords_01_atom_01[1518], [52.372, 15.397, -15.323]);

    let atom01 = mols[0].get_atom(0);
    assert!(atom01.x() == 50.87300109863281);
    assert!(atom01.y() == 32.97800064086914);
    assert!(atom01.z() == 2.38700008392334);

    let chains = mols[0].get_chains();
    let residues = mols[0].get_residues_by_chain(chains[0].clone());
    let residue = mols[0].create_residue(chains[0].clone(), residues[0]);
    assert!(residue.name() == Some("VAL"));

    // Original
    // ATOM      1  N   VAL A   1      50.873  32.978   2.387  1.00 27.72      A    N
    // ATOM  1     N    VAL A1         50.873  32.978   2.387  0.00 27.72          N

    // Original
    // MASTER      365    0    0    5   18    0    0    6 1519    1    0   15
    // MASTER    0    0    0    0    0    0    0    6    1519 1    0    0

    let chain = mols[0].create_chain(chains[0].clone());

    let view = &psedata.view;
    println!("{:?}", view);

    // Check symmetry code
    let (unit, sym) = mols[0].get_unit_cell_symmetry();

    // Check the pymol object fields
    let pyobj = &mols[0].object;
    let color = &pyobj.get_color();
    assert_eq!(
        *color,
        Color {
            name: "carbon",
            r: 0.2,
            g: 1.0,
            b: 0.2
        }
    );

    let vis_rep = &pyobj.vis_rep;
    assert!(vis_rep.contains(RepBitmask::CYL));
    // check the name
    assert_eq!(&pyobj.name, "1pdb");

    // Move on to PDB, baby!
    let pdb = mols[0].to_pdb();

    let _ = pdbtbx::save_pdb(
        &pdb,
        format!("{}/test_01.pdb", TEST_OUTPUT_DIR),
        pdbtbx::StrictnessLevel::Strict,
    )
    .expect("PDB output");
}

#[test]
fn test_pdb_01() {
    let psedata: PSEData = PSEData::load("tests/data/example.pse").unwrap();
    // let _ = psedata.to_disk(TEST_OUTPUT_DIR);
    let _ = psedata.to_disk_full(TEST_OUTPUT_DIR);
    let url = psedata.to_mvsj_url();
    println!("{}", url);
}

#[test]
fn test_colors() {
    // https://github.com/schrodinger/pymol-open-source/blob/master/layer1/Color.cpp#L880
    let spectrum_s: [[f32; 3]; 13] = [
        [1.0, 0.0, 1.0], // magenta - 0
        [0.5, 0.0, 1.0],
        [0.0, 0.0, 1.0], // blue - 166.66
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 1.0], // cyan - 333.33
        [0.0, 1.0, 0.5],
        [0.0, 1.0, 0.0], // green - 500
        [0.5, 1.0, 0.0],
        [1.0, 1.0, 0.0], // yellow - 666.66
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 0.0], // red - 833.33
        [1.0, 0.0, 0.5],
        [1.0, 0.0, 1.0], // magenta - 999
    ];

    const A_DIV: f32 = 83.333333333;

    // Full spectrum (s000-s999)
    for a in 0..1000 {
        let set1 = (a as f32 / A_DIV) as usize;
        let name = format!("s{:03}", a);
        let f = 1.0 - (a as f32 - (set1 as f32 * A_DIV)) / A_DIV;
        let r = f * spectrum_s[set1][0] + (1.0 - f) * spectrum_s[set1 + 1][0];
        let g = f * spectrum_s[set1][1] + (1.0 - f) * spectrum_s[set1 + 1][1];
        let b = f * spectrum_s[set1][2] + (1.0 - f) * spectrum_s[set1 + 1][2];
        println!(
            " Color {{ name: \"{}\", r: {:?}, g: {:?}, b: {:?} }},",
            name, r, g, b
        );
    }

    let spectrum_r: [[f32; 3]; 13] = [
        [1.0, 1.0, 0.0], // yellow - 0
        [0.5, 1.0, 0.0], // chartreuse
        [0.0, 1.0, 0.0], // green - 166.66
        [0.0, 1.0, 0.5], // limegreen
        [0.0, 1.0, 1.0], // cyan - 333.33
        [0.0, 0.5, 1.0], // marine
        [0.0, 0.0, 1.0], // blue - 500
        [0.5, 0.0, 1.0], // purpleblue
        [1.0, 0.0, 1.0], // magenta - 666.66
        [1.0, 0.0, 0.5], // hotpink
        [1.0, 0.0, 0.0], // red - 833.33
        [1.0, 0.5, 0.0], // orange
        [1.0, 1.0, 0.0], // yellow - 999
    ];

    for a in 0..1000 {
        let set1 = (a as f32 / A_DIV) as usize;
        let name = format!("r{:03}", a);
        let f = 1.0 - (a as f32 - (set1 as f32 * A_DIV)) / A_DIV;
        let r = f * spectrum_r[set1][0] + (1.0 - f) * spectrum_r[set1 + 1][0];
        let g = f * spectrum_r[set1][1] + (1.0 - f) * spectrum_r[set1 + 1][1];
        let b = f * spectrum_r[set1][2] + (1.0 - f) * spectrum_r[set1 + 1][2];
        // Assuming reg_named_color is implemented elsewhere
        println!(
            " Color {{ name: \"{}\", r: {:?}, g: {:?}, b: {:?} }},",
            name, r, g, b
        );
    }

    let spectrum_c: [[f32; 3]; 13] = [
        [1.0, 1.0, 0.0], // yellow - 0
        [0.0, 0.0, 1.0], // blue - 83.333
        [1.0, 0.0, 0.0], // red - 167.67
        [0.0, 1.0, 0.0], // green - 250.00
        [1.0, 0.0, 1.0], // magenta - 333.33
        [0.0, 1.0, 1.0], // cyan - 416.67
        [1.0, 1.0, 0.0], // yellow - 500.00
        [0.0, 1.0, 0.0], // green - 583.33
        [0.0, 0.0, 1.0], // blue - 666.67
        [1.0, 0.0, 1.0], // magenta - 750.00
        [1.0, 1.0, 0.0], // yellow - 833.33
        [1.0, 0.0, 0.0], // red - 916.67
        [0.0, 1.0, 1.0], // cyan - 999
    ];

    for a in 0..1000 {
        let set1 = (a as f32 / A_DIV) as usize;
        // sprintf(color->Name,"c%03d",a);
        let name = format!("c{:03}", a);
        let f = 1.0 - (a as f32 - (set1 as f32 * A_DIV)) / A_DIV;
        let r = f * spectrum_c[set1][0] + (1.0 - f) * spectrum_c[set1 + 1][0];
        let g = f * spectrum_c[set1][1] + (1.0 - f) * spectrum_c[set1 + 1][1];
        let b = f * spectrum_c[set1][2] + (1.0 - f) * spectrum_c[set1 + 1][2];
        println!(
            " Color {{ name: \"{}\", r: {:?}, g: {:?}, b: {:?} }},",
            name, r, g, b
        );
    }

    let spectrum_w: [[f32; 3]; 25] = [
        [1.0, 1.0, 0.0], // yellow - 0
        [1.0, 1.0, 1.0], // white
        [0.0, 0.0, 1.0], // blue  - 83.333
        [1.0, 1.0, 1.0], // white
        [1.0, 0.0, 0.0], // red - 166.67
        [1.0, 1.0, 1.0], // white
        [0.0, 1.0, 0.0], // green - 250.00
        [1.0, 1.0, 1.0], // white
        [1.0, 0.0, 1.0], // magenta - 333.33
        [1.0, 1.0, 1.0], // white
        [0.0, 1.0, 1.0], // cyan - 416.67
        [1.0, 1.0, 1.0], // white
        [1.0, 1.0, 0.0], // yellow - 500.00
        [1.0, 1.0, 1.0], // white
        [0.0, 1.0, 0.0], // green - 583.33
        [1.0, 1.0, 1.0], // white
        [0.0, 0.0, 1.0], // blue - 666.67
        [1.0, 1.0, 1.0], // white
        [1.0, 0.0, 1.0], // magenta - 750.00
        [1.0, 1.0, 1.0], // white
        [1.0, 1.0, 0.0], // yellow - 833.33
        [1.0, 1.0, 1.0], // white
        [1.0, 0.0, 0.0], // red - 916.67
        [1.0, 1.0, 1.0], // white
        [0.0, 1.0, 1.0], // cyan - 999
    ];

    const W_DIV: f32 = 41.666666667;

    // complementary spectra separated by white (w000-w999)

    for a in 0..1000 {
        let set1 = (a as f32 / W_DIV) as usize;
        let name = format!("w{:03}", a);
        let f = 1.0 - (a as f32 - (set1 as f32 * W_DIV)) / W_DIV;
        let r = f * spectrum_w[set1][0] + (1.0 - f) * spectrum_w[set1 + 1][0];
        let g = f * spectrum_w[set1][1] + (1.0 - f) * spectrum_w[set1 + 1][1];
        let b = f * spectrum_w[set1][2] + (1.0 - f) * spectrum_w[set1 + 1][2];
        println!(
            " Color {{ name: \"{}\", r: {:?}, g: {:?}, b: {:?} }},",
            name, r, g, b
        );
    }

    for a in 0..100 {
        let name = format!("gray{:02}", a);
        let value = a as f32 / 99.0;
        println!(
            " Color {{ name: \"{}\", r: {:?}, g: {:?}, b: {:?} }},",
            name, value, value, value
        );
    }

    let spectrum_o: [[f32; 3]; 29] = [
        /* a rainbow with perceptive color balancing and extra blue/red at the ends */
        [1.0, 0.0, 1.0], /* violet */
        [0.8, 0.0, 1.0],
        [0.5, 0.0, 1.0], /* blend */
        [0.0, 0.0, 1.0], /* blue */
        [0.0, 0.0, 1.0], /* blue */
        [0.0, 0.2, 1.0],
        [0.0, 0.5, 1.0], /* blend */
        [0.0, 0.8, 1.0],
        [0.0, 1.0, 1.0], /* cyan */
        [0.0, 1.0, 0.8],
        [0.0, 1.0, 0.5], /* blend */
        [0.0, 1.0, 0.2],
        [0.0, 1.0, 0.0], /* green */
        [0.2, 1.0, 0.0],
        [0.5, 1.0, 0.0], /* blend */
        [0.8, 1.0, 0.0],
        [1.0, 1.0, 0.0], /* yellow */
        [1.0, 0.9, 0.0],
        [1.0, 0.75, 0.0], /* blend */
        [1.0, 0.6, 0.0],
        [1.0, 0.5, 0.0], /* orange */
        [1.0, 0.4, 0.0],
        [1.0, 0.3, 0.0], /* blend */
        [1.0, 0.2, 0.0],
        [1.0, 0.0, 0.0], /* red */
        [1.0, 0.0, 0.0], /* red */
        [1.0, 0.0, 0.5], /* blend */
        [1.0, 0.0, 0.8], /* violet */
        [1.0, 0.0, 1.0], /* violet */
    ];

    const B_DIV: f32 = 35.7143;

    for a in 0..1000 {
        let set1 = (a as f32 / B_DIV) as usize;
        let name = format!("o{:03}", a);
        let f = 1.0 - (a as f32 - (set1 as f32 * B_DIV)) / B_DIV;
        let r = f * spectrum_o[set1][0] + (1.0 - f) * spectrum_o[set1 + 1][0];
        let g = f * spectrum_o[set1][1] + (1.0 - f) * spectrum_o[set1 + 1][1];
        let b = f * spectrum_o[set1][2] + (1.0 - f) * spectrum_o[set1 + 1][2];
        println!(
            " Color {{ name: \"{}\", r: {:?}, g: {:?}, b: {:?} }},",
            name, r, g, b
        );
    }
}
