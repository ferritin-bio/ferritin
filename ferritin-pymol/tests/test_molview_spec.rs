use pseutils::molviewspec::nodes::{
    ColorT, ComponentExpression, ComponentSelector, ComponentSelectorT, ParseFormatT, ParseParams,
    RepresentationTypeT, State, StructureParams, StructureTypeT,
};
use serde_json::from_reader;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;

const TEST_OUTPUT_DIR: &str = "./test_temporary";

#[test]
fn test_molspecview_json_1cbs() {
    let json_files_component_list = vec![
        "tests/mol-spec-data/1cbs/auth_residue.json",
        "tests/mol-spec-data/1cbs/chain_label.json",
        "tests/mol-spec-data/1cbs/rainbow.json",
        "tests/mol-spec-data/1cbs/validation.json",
    ];

    for json_file in json_files_component_list {
        let file = File::open(json_file).expect(&format!("Failed to open file: {}", json_file));
        let reader = BufReader::new(file);
        let _: Vec<ComponentExpression> = from_reader(reader).expect(&format!(
            "Failed to parse JSON as a Vector of ComponentExpressions: {}",
            json_file
        ));
    }

    // // Todo: Fix the Resideu_range_component/////
    // let json_files_component = vec!["tests/mol-spec-data/1cbs/auth_residue_range.json"];
    // for json_file in json_files_component {
    //     let file = File::open(json_file).expect(&format!("Failed to open file: {}", json_file));
    //     let reader = BufReader::new(file);
    //     let _: ComponentExpression = from_reader(reader).expect(&format!(
    //         "Failed to parse JSON as ComponentExpression: {}",
    //         json_file
    //     ));
    // }
}

#[test]
fn test_molspecview_json_1h9t() {
    let file = File::open("tests/mol-spec-data/1h9t/domains.json").expect("Failed to open file");
    let reader = BufReader::new(file);
    let testvec: Vec<ComponentExpression> =
        from_reader(reader).expect("Failed to parse JSON as ComponentExpression");

    assert_eq!(testvec[0].label_asym_id, Some("A".to_string()));
    assert_eq!(testvec[0].beg_label_seq_id, Some(9));
    assert_eq!(testvec[0].end_label_seq_id, Some(83));

    // todo: these shouw work!
    // assert_eq!(testvec[0].color, "#dd6600");
    // assert_eq!(testvec[0].tooltip, "DNA-binding");

    // let file = File::open("tests/mol-spec-data/1h9t/domains.json").expect("Failed to open file");
    // let reader = BufReader::new(file);
    // let testvec: Vec<ComponentExpression> =
    //     from_reader(reader).expect("Failed to parse JSON as ComponentExpression");
}

#[test]
fn test_molspecview_json_2bvk() {
    let file = File::open("tests/mol-spec-data/2bvk/atoms.json").expect("Failed to open file");
    let reader = BufReader::new(file);
    let testvec: Vec<ComponentExpression> =
        from_reader(reader).expect("Failed to parse JSON as ComponentExpression");

    assert_eq!(testvec[0].atom_index, Some(0));
}

#[test]
fn test_moviewspec_00_builder_basics() {
    // https://colab.research.google.com/drive/1O2TldXlS01s-YgkD9gy87vWsfCBTYuz9#scrollTo=U256gC0Tj2vS
    // builder = mvs.create_builder()
    // (
    //     builder.download(url='https://files.wwpdb.org/download/1cbs.cif')
    //     .parse(format='mmcif')
    //     .assembly_structure(assembly_id='1')
    //     .component()
    //     .representation()
    // )

    // parse params
    let structfile = ParseParams {
        format: ParseFormatT::Mmcif,
    };

    // struct params
    let structparams = StructureParams {
        structure_type: StructureTypeT::Assembly,
        assembly_id: Some('1'.to_string()),
        ..Default::default()
    };

    // define the component which is going to be `all` here
    let component = ComponentSelector::default();
    let cartoon_type = RepresentationTypeT::Cartoon;
    let mut state = State::new();
    state
        .download("https://files.wwpdb.org/download/1cbs.cif")
        .expect("Create a Downlaod node with a URL")
        .parse(structfile)
        .expect("Parseable option")
        .assembly_structure(structparams)
        .expect("a set of Structure options")
        .component(component)
        .expect("defined a valid component")
        .representation(cartoon_type);
    // .expect("a valid representation")
    // .color(color, color_component);
    std::fs::create_dir_all(TEST_OUTPUT_DIR).expect("Failed to create output directory");
    let pretty_json = serde_json::to_string_pretty(&state).unwrap();
    let mut file = File::create(format!("{}/test_moviewspec_01.json", TEST_OUTPUT_DIR)).unwrap();
    file.write_all(pretty_json.as_bytes()).unwrap();
}

#[test]
fn test_moviewspec_01_common_actions_cartoon() {
    // https://colab.research.google.com/drive/1O2TldXlS01s-YgkD9gy87vWsfCBTYuz9#scrollTo=U256gC0Tj2vS
    // builder = mvs.create_builder()
    // (
    //     builder.download(url='https://files.wwpdb.org/download/1cbs.cif')
    //     .parse(format='mmcif')
    //     .assembly_structure(assembly_id='1')
    //     .component()
    //     .representation()
    //     .color(color='#1b9e77')
    // )

    // parse params
    let structfile = ParseParams {
        format: ParseFormatT::Mmcif,
    };

    // struct params
    let structparams = StructureParams {
        structure_type: StructureTypeT::Assembly,
        assembly_id: Some('1'.to_string()),
        ..Default::default()
    };

    // define the component which is going to be `all` here
    let component = ComponentSelector::default();
    // cartoon type
    let cartoon_type = RepresentationTypeT::Cartoon;

    // color
    let color = ColorT::Hex("#1b9e77".to_string());
    let color_component = ComponentSelector::default();

    let mut state = State::new();

    state
        .download("https://files.wwpdb.org/download/1cbs.cif")
        .expect("Create a Downlaod node with a URL")
        .parse(structfile)
        .expect("Parseable option")
        .assembly_structure(structparams)
        .expect("a set of Structure options")
        .component(component)
        .expect("defined a valid component")
        .representation(cartoon_type)
        .expect("a valid representation")
        .color(color, color_component);

    let pretty_json = serde_json::to_string_pretty(&state).unwrap();
    let mut file = File::create(format!(
        "{}/test_moviewspec_01_common_actions_cartoon.json",
        TEST_OUTPUT_DIR
    ))
    .unwrap();
    file.write_all(pretty_json.as_bytes()).unwrap();
}

#[test]
fn test_moviewspec_01_common_actions_selectors() {
    // https://colab.research.google.com/drive/1O2TldXlS01s-YgkD9gy87vWsfCBTYuz9#scrollTo=U256gC0Tj2vS
    // builder = mvs.create_builder()
    // structure = builder.download(url="https://files.wwpdb.org/download/1c0a.cif").parse(format="mmcif").assembly_structure()
    // # represent protein & RNA as cartoon
    // structure.component(selector="protein").representation().color(color="#e19039")  # protein in orange
    // structure.component(selector="nucleic").representation().color(color="#4b7fcc")  # RNA in blue
    // # represent ligand in active site as ball-and-stick
    // ligand = structure.component(selector=mvs.ComponentExpression(label_asym_id='E'))
    // ligand.representation(type="ball_and_stick").color(color="#229954")  # ligand in green
    // # represent 2 crucial arginine residues as red ball-and-stick and label with custom text
    // arg_b_217 = structure.component(selector=mvs.ComponentExpression(label_asym_id="B", label_seq_id=217))
    // arg_b_217.representation(type="ball_and_stick").color(color="#ff0000")
    // arg_b_217.label(text="aaRS Class II Signature")
    // arg_b_537 = structure.component(selector=mvs.ComponentExpression(label_asym_id="B", label_seq_id=537))
    // arg_b_537.representation(type="ball_and_stick").color(color="#ff0000")
    // arg_b_537.label(text="aaRS Class II Signature")

    // # position camera to zoom in on ligand and signature residues
    // focus = structure.component(selector=[mvs.ComponentExpression(label_asym_id='E'), mvs.ComponentExpression(label_asym_id="B", label_seq_id=217), mvs.ComponentExpression(label_asym_id="B", label_seq_id=537)]).focus()
    // parse params
    let structfile = ParseParams {
        format: ParseFormatT::Mmcif,
    };
    // struct params
    let structparams = StructureParams {
        structure_type: StructureTypeT::Assembly,
        ..Default::default()
    };
    // State is the base model
    let mut state = State::new();
    let structure = state
        .download("https://files.wwpdb.org/download/1cbs.cif")
        .expect("Create a Download node with a URL")
        .parse(structfile)
        .expect("Parseable option")
        .assembly_structure(structparams)
        .expect("Structure params");
    let orange = ColorT::Hex("#e19039".to_string());
    let blue = ColorT::Hex("#4b7fcc".to_string());
    let green = ColorT::Hex("#229954".to_string());
    let dark = ColorT::Hex("#ff0000".to_string());
    // set protein as orange
    let component_prot = ComponentSelector::Selector(ComponentSelectorT::Protein);
    structure
        .component(component_prot)
        .expect("Created component")
        .representation(RepresentationTypeT::Cartoon)
        .expect("Faithful representation")
        .color(
            orange,
            ComponentSelector::Selector(ComponentSelectorT::Protein),
        );

    // set RNA as blue
    let component_rna = ComponentSelector::Selector(ComponentSelectorT::Nucleic);
    structure
        .component(component_rna)
        .expect("Created component")
        .representation(RepresentationTypeT::Cartoon)
        .expect("Faithful representation")
        .color(blue, ComponentSelector::default());

    // ligand is green
    let ligand = structure
        .component(ComponentSelector::Expression(ComponentExpression {
            label_asym_id: Some("E".to_string()),
            ..Default::default()
        }))
        .expect("Expectation");

    ligand
        .representation(RepresentationTypeT::Cartoon)
        .expect("Represented correctly")
        .color(green, ComponentSelector::default());

    let arg_b_217 = structure
        .component(ComponentSelector::Expression(ComponentExpression {
            label_asym_id: Some("B".to_string()),
            label_seq_id: Some(217),
            ..Default::default()
        }))
        .expect("Expectation");

    arg_b_217
        .representation(RepresentationTypeT::BallAndStick)
        .expect("Representation")
        .color(dark.clone(), ComponentSelector::default())
        .expect("out");

    arg_b_217.label("aaRS Class II Signature".to_string());

    let arg_b_537 = structure
        .component(ComponentSelector::Expression(ComponentExpression {
            label_asym_id: Some("B".to_string()),
            label_seq_id: Some(537),
            ..Default::default()
        }))
        .expect("Expectation");

    arg_b_537
        .representation(RepresentationTypeT::BallAndStick)
        .expect("Representation")
        .color(dark.clone(), ComponentSelector::default())
        .expect("out");

    arg_b_537.label("aaRS Class II Signature".to_string());

    // Todo: implement focus
    // focus = structure.component(selector=[mvs.ComponentExpression(label_asym_id='E'), mvs.ComponentExpression(label_asym_id="B", label_seq_id=217), mvs.ComponentExpression(label_asym_id="B", label_seq_id=537)]).focus()

    let pretty_json = serde_json::to_string_pretty(&state).unwrap();
    let mut file = File::create(format!(
        "{}/test_moviewspec_01_common_actions_selectors.json",
        TEST_OUTPUT_DIR
    ))
    .unwrap();
    file.write_all(pretty_json.as_bytes()).unwrap();
}

#[test]
fn test_moviewspec_01_common_actions_symmetry() {
    // https://colab.research.google.com/drive/1O2TldXlS01s-YgkD9gy87vWsfCBTYuz9#scrollTo=U256gC0Tj2vS
    // builder = mvs.create_builder()
    // builder = mvs.create_builder()    // (
    //     builder.download(url="https://files.wwpdb.org/download/4hhb.cif")
    //     .parse(format="mmcif")
    //     .symmetry_mates_structure(radius=5.0)
    //     .component()
    //     .representation()
    //     .color(color='#1b9e77')

    // struct params
    let structparams = StructureParams {
        structure_type: StructureTypeT::Symmetry, // <----- Main difference here
        radius: Some(5.0),
        ..Default::default()
    };

    // color
    let color = ColorT::Hex("#1b9e77".to_string());
    let color_component = ComponentSelector::default();

    let mut state = State::new();
    state
        .download("https://files.wwpdb.org/download/4hhb.cif")
        .expect("Create a Downlaod node with a URL")
        .parse(ParseParams {
            format: ParseFormatT::Mmcif,
        })
        .expect("Parseable option")
        .symmetry_mates_structure(structparams)
        .expect("a set of Structure options")
        .component(ComponentSelector::default())
        .expect("defined a valid component")
        .representation(RepresentationTypeT::Cartoon)
        .expect("a valid representation")
        .color(color, color_component);

    let pretty_json = serde_json::to_string_pretty(&state).unwrap();
    let mut file = File::create(format!(
        "{}/test_moviewspec_01_common_actions_cartoon.json",
        TEST_OUTPUT_DIR
    ))
    .unwrap();
    file.write_all(pretty_json.as_bytes()).unwrap();
}

#[test]
fn test_moviewspec_01_common_actions_symmetry_miller() {
    // https://colab.research.google.com/drive/1O2TldXlS01s-YgkD9gy87vWsfCBTYuz9#scrollTo=U256gC0Tj2vS
    // builder = mvs.create_builder()
    //     builder.download(url="https://files.wwpdb.org/download/4hhb.cif")
    //     .parse(format="mmcif")
    //     .symmetry_structure(ijk_min=(-1, -1, -1), ijk_max=(1, 1, 1))
    //     .component()
    //     .representation()
    //     .color(color='#1b9e77')

    // struct params
    let structparams = StructureParams {
        structure_type: StructureTypeT::SymmetryMates, // <----- Main difference here
        ijk_min: Some((-1, -1, -1)),
        ijk_max: Some((1, 1, 1)),
        ..Default::default()
    };

    // color
    let color = ColorT::Hex("#1b9e77".to_string());
    let color_component = ComponentSelector::default();

    let mut state = State::new();

    state
        .download("https://files.wwpdb.org/download/4hhb.cif")
        .expect("Create a Downlaod node with a URL")
        .parse(ParseParams {
            format: ParseFormatT::Mmcif,
        })
        .expect("Parseable option")
        .symmetry_mates_structure(structparams)
        .expect("a set of Structure options")
        .component(ComponentSelector::default())
        .expect("defined a valid component")
        .representation(RepresentationTypeT::Cartoon)
        .expect("a valid representation")
        .color(color, color_component);

    let pretty_json = serde_json::to_string_pretty(&state).unwrap();
    let mut file = File::create(format!(
        "{}/test_moviewspec_01_common_actions_symmetry_miller.json",
        TEST_OUTPUT_DIR
    ))
    .unwrap();
    file.write_all(pretty_json.as_bytes()).unwrap();
    println!("{}", state.to_url())
}

#[test]
fn test_moviewspec_01_common_actions_transform_superimpose() {
    // builder = mvs.create_builder()
    // structure1 = (
    //     builder.download(url="https://files.wwpdb.org/download/1oj6.cif")
    //     .parse(format="mmcif")
    //     .assembly_structure()
    // )
    // # 1st structure colored in orange
    // structure1.component(selector='polymer').representation(type='cartoon').color(color='#e19039')
    // structure1.component(selector='ligand').representation(type='ball_and_stick').color(color='#eec190')
    //
    // structure2 = (
    //     builder.download(url="https://files.wwpdb.org/download/5mjd.cif")
    //     .parse(format="mmcif")
    //     .assembly_structure()
    //     # move these coordinates to align both structures
    //     .transform(
    //         rotation=[-0.39652203922082313, 0.918022802798312, 0.002099036562725462, 0.9068461182538327, 0.39133670281585825, 0.1564790811487865, 0.14282993460796656, 0.06395090751149791, -0.9876790426086504],
    //         translation=[-17.636085896690037, 7.970761314734439, 88.54613248028247]
    //     )
    // )
    // # 2nd structure colored in blue
    // structure2.component(selector='polymer').representation(type='cartoon').color(color='#4b7fcc')
    // structure2.component(selector='ligand').representation(type='ball_and_stick').color(color='#9cb8e3')
    // print(builder.get_state())

    //Todo
}
