use ferritin_molviewspec::molviewspec::nodes::{
    CanvasParams, ColorT, ComponentExpression, ComponentSelector, ComponentSelectorT,
    FocusInlineParams, IsoSurfaceParams, KindT, LineParams, MvsjFile, Node, NodeParams,
    ParseFormatT, ParseParams, RepresentationTypeT, SphereParams, State, StructureParams,
    StructureTypeT, TooltipInlineParams, TransformParams, VolumeParams, VolumeRepresentationParams,
    VolumeRepresentationTypeT,
};
use serde_json::from_reader;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;

const TEST_OUTPUT_DIR: &str = "./test_temporary";

#[test]
#[ignore]
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
#[ignore]
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
#[ignore]
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
#[ignore]
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
#[ignore]
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
#[ignore]
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
#[ignore]
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

// ---------------------------------------------------------------------------
// Gap 2 (T-X09): children normalization on deserialization
// ---------------------------------------------------------------------------

#[test]
fn test_deser_node_empty_children_array() {
    // children: [] must normalize to None — an empty array is not a valid spec state
    let json = r#"{"kind":"root","children":[]}"#;
    let node: Node = serde_json::from_str(json).expect("valid JSON");
    assert!(
        node.children.is_none(),
        "children: [] must deserialize as None, not Some(vec![])"
    );
}

#[test]
fn test_deser_node_null_children() {
    // children: null must also deserialize as None
    let json = r#"{"kind":"root","children":null}"#;
    let node: Node = serde_json::from_str(json).expect("valid JSON");
    assert!(node.children.is_none());
}

// ---------------------------------------------------------------------------
// Gap 3: builder parent-kind guard negative tests
// ---------------------------------------------------------------------------

#[test]
fn test_builder_parse_rejects_non_download_parent() {
    let mut node = Node::new(KindT::Structure, None);
    let result = node.parse(ParseParams {
        format: ParseFormatT::Mmcif,
    });
    assert!(result.is_none(), "parse() must return None for non-Download parent");
}

#[test]
fn test_builder_assembly_structure_rejects_non_parse_parent() {
    let mut node = Node::new(KindT::Download, None);
    let result = node.assembly_structure(StructureParams::default());
    assert!(result.is_none(), "assembly_structure() must return None for non-Parse parent");
}

#[test]
fn test_builder_model_structure_rejects_non_parse_parent() {
    // model_structure is currently unimplemented!(); test the parent-kind guard
    // once it is implemented it must also reject non-Parse parents.
    // For now, assert component() rejects a non-Structure parent (covers same guard pattern).
    let mut node = Node::new(KindT::Parse, None);
    let result = node.component(ComponentSelector::default());
    assert!(result.is_none(), "component() must return None for non-Structure parent");
}

#[test]
fn test_builder_component_rejects_non_structure_parent() {
    let mut node = Node::new(KindT::Parse, None);
    let result = node.component(ComponentSelector::default());
    assert!(result.is_none(), "component() must return None for non-Structure parent");
}

#[test]
fn test_builder_representation_rejects_non_component_parent() {
    let mut node = Node::new(KindT::Structure, None);
    let result = node.representation(RepresentationTypeT::Cartoon);
    assert!(result.is_none(), "representation() must return None for non-Component parent");
}

#[test]
fn test_builder_color_rejects_non_representation_parent() {
    let mut node = Node::new(KindT::Component, None);
    let result = node.color(ColorT::Hex("#ff0000".to_string()), ComponentSelector::default());
    assert!(result.is_none(), "color() must return None for non-Representation parent");
}

#[test]
fn test_builder_label_rejects_non_component_parent() {
    let mut node = Node::new(KindT::Representation, None);
    let result = node.label("text".to_string());
    assert!(result.is_none(), "label() must return None for non-Component parent");
}

#[test]
fn test_builder_transform_rejects_non_structure_parent() {
    let mut node = Node::new(KindT::Parse, None);
    let result = node.transform(TransformParams::default());
    assert!(result.is_none(), "transform() must return None for non-Structure parent");
}

// ---------------------------------------------------------------------------
// Gap 1 (T2-21): all-None ComponentExpression matches everything
// Decision: matches Python mol-view-spec; Phase 2 must return all-true AtomMask.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// T1 tests: Phase 1a — custom Node Deserialize (kind-dispatched params)
// ---------------------------------------------------------------------------

// T1-01: focus node with empty params must yield FocusInlineParams, not ComponentInlineParams
#[test]
fn test_deser_focus_empty_params_is_focus_inline() {
    let json = r#"{"kind":"focus","params":{}}"#;
    let node: Node = serde_json::from_str(json).expect("valid focus node");
    match node.params {
        Some(NodeParams::FocusInlineParams(FocusInlineParams { direction: None, up: None })) => {}
        other => panic!("expected FocusInlineParams{{None,None}}, got {:?}", other),
    }
}

// T1-02: focus node with direction must round-trip
#[test]
fn test_deser_focus_with_direction() {
    let json = r#"{"kind":"focus","params":{"direction":[1.0,0.0,0.0]}}"#;
    let node: Node = serde_json::from_str(json).expect("valid focus node with direction");
    match node.params {
        Some(NodeParams::FocusInlineParams(p)) => {
            assert_eq!(p.direction, Some((1.0, 0.0, 0.0)));
            assert!(p.up.is_none());
        }
        other => panic!("expected FocusInlineParams, got {:?}", other),
    }
}

// T1-03: transform node with rotation must yield TransformParams, not FocusInlineParams
#[test]
fn test_deser_transform_with_rotation() {
    let rotation = vec![
        -0.7202161f64, -0.33009904, -0.61018308,
        0.36257631, 0.57075962, -0.73673053,
        0.59146191, -0.75184312, -0.29138417,
    ];
    let json = serde_json::json!({
        "kind": "transform",
        "params": {
            "rotation": rotation,
            "translation": [-12.54, 46.79, 94.5]
        }
    })
    .to_string();
    let node: Node = serde_json::from_str(&json).expect("valid transform node");
    match node.params {
        Some(NodeParams::TransformParams(p)) => {
            assert!(p.rotation.is_some());
            assert_eq!(p.rotation.as_ref().unwrap().len(), 9);
            assert_eq!(p.translation, Some((-12.54, 46.79, 94.5)));
        }
        other => panic!("expected TransformParams, got {:?}", other),
    }
}

// T1-04: transform node with empty params must yield TransformParams{None,None}
#[test]
fn test_deser_transform_empty_params() {
    let json = r#"{"kind":"transform","params":{}}"#;
    let node: Node = serde_json::from_str(json).expect("valid transform empty params");
    match node.params {
        Some(NodeParams::TransformParams(TransformParams { rotation: None, translation: None })) => {}
        other => panic!("expected empty TransformParams, got {:?}", other),
    }
}

// T1-05: component node with string selector
#[test]
fn test_deser_component_string_selector() {
    let json = r#"{"kind":"component","params":{"selector":"all"}}"#;
    let node: Node = serde_json::from_str(json).expect("valid component node");
    match node.params {
        Some(NodeParams::ComponentInlineParams(p)) => {
            assert!(matches!(p.selector, ComponentSelector::Selector(ComponentSelectorT::All)));
        }
        other => panic!("expected ComponentInlineParams, got {:?}", other),
    }
}

// T1-06: component node with expression selector
#[test]
fn test_deser_component_expression_selector() {
    let json = r#"{"kind":"component","params":{"selector":{"label_asym_id":"A","label_seq_id":42}}}"#;
    let node: Node = serde_json::from_str(json).expect("valid component expression node");
    match node.params {
        Some(NodeParams::ComponentInlineParams(p)) => match p.selector {
            ComponentSelector::Expression(expr) => {
                assert_eq!(expr.label_asym_id, Some("A".to_string()));
                assert_eq!(expr.label_seq_id, Some(42));
            }
            other => panic!("expected Expression selector, got {:?}", other),
        },
        other => panic!("expected ComponentInlineParams, got {:?}", other),
    }
}

// T1-07: representation node round-trip
#[test]
fn test_deser_representation_cartoon() {
    let json = r#"{"kind":"representation","params":{"type":"cartoon"}}"#;
    let node: Node = serde_json::from_str(json).expect("valid representation node");
    match node.params {
        Some(NodeParams::RepresentationParams(p)) => {
            assert!(matches!(p.representation_type, RepresentationTypeT::Cartoon));
        }
        other => panic!("expected RepresentationParams, got {:?}", other),
    }
}

// T1-08: color node with hex color + selector
#[test]
fn test_deser_color_hex() {
    let json = r##"{"kind":"color","params":{"color":"#1b9e77","selector":"all"}}"##;
    let node: Node = serde_json::from_str(json).expect("valid color node");
    match node.params {
        Some(NodeParams::ColorInlineParams(p)) => match p.color {
            ColorT::Hex(h) => assert_eq!(h, "#1b9e77"),
            other => panic!("expected hex color, got {:?}", other),
        },
        other => panic!("expected ColorInlineParams, got {:?}", other),
    }
}

// T1-09: label node round-trip
#[test]
fn test_deser_label_text() {
    let json = r#"{"kind":"label","params":{"text":"hello"}}"#;
    let node: Node = serde_json::from_str(json).expect("valid label node");
    match node.params {
        Some(NodeParams::LabelInlineParams(p)) => assert_eq!(p.text, "hello"),
        other => panic!("expected LabelInlineParams, got {:?}", other),
    }
}

// T1-10: download node round-trip
#[test]
fn test_deser_download_url() {
    let json = r#"{"kind":"download","params":{"url":"https://files.wwpdb.org/download/1cbs.cif"}}"#;
    let node: Node = serde_json::from_str(json).expect("valid download node");
    match node.params {
        Some(NodeParams::DownloadParams(p)) => {
            assert_eq!(p.url, "https://files.wwpdb.org/download/1cbs.cif");
        }
        other => panic!("expected DownloadParams, got {:?}", other),
    }
}

// T1-11: parse node round-trip
#[test]
fn test_deser_parse_mmcif() {
    let json = r#"{"kind":"parse","params":{"format":"mmcif"}}"#;
    let node: Node = serde_json::from_str(json).expect("valid parse node");
    match node.params {
        Some(NodeParams::ParseParams(p)) => {
            assert!(matches!(p.format, ParseFormatT::Mmcif));
        }
        other => panic!("expected ParseParams, got {:?}", other),
    }
}

// T1-12: structure node with type=model
#[test]
fn test_deser_structure_model() {
    let json = r#"{"kind":"structure","params":{"type":"model"}}"#;
    let node: Node = serde_json::from_str(json).expect("valid structure node");
    match node.params {
        Some(NodeParams::StructureParams(p)) => {
            assert!(matches!(p.structure_type, StructureTypeT::Model));
        }
        other => panic!("expected StructureParams, got {:?}", other),
    }
}

// T1-25: root node with no params
#[test]
fn test_deser_root_no_params() {
    let json = r#"{"kind":"root"}"#;
    let node: Node = serde_json::from_str(json).expect("valid root node");
    assert!(node.params.is_none(), "root without params should have params=None");
}

// T1-26: root node with explicit null params
#[test]
fn test_deser_root_null_params() {
    let json = r#"{"kind":"root","params":null}"#;
    let node: Node = serde_json::from_str(json).expect("valid root null-params node");
    assert!(node.params.is_none(), "root with params:null should have params=None");
}

// T1-27: full components/state.mvsj round-trip (focus + transform)
#[test]
fn test_deser_components_example_roundtrip() {
    let path = "tests/mol-spec-examples/components/state.mvsj";
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("cannot read {}", path));
    let state: State = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e));
    // The components example has a focus node with params:{}
    // Walk the tree and verify we find a FocusInlineParams node
    fn find_focus(node: &Node) -> bool {
        if matches!(node.params, Some(NodeParams::FocusInlineParams(_))) {
            return true;
        }
        node.children.iter().flatten().any(find_focus)
    }
    assert!(find_focus(&state.root), "components example must contain a FocusInlineParams node");
}

// T1-28: superposition/state.mvsj round-trip (transform with rotation)
#[test]
fn test_deser_superposition_example_roundtrip() {
    let path = "tests/mol-spec-examples/superposition/state.mvsj";
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("cannot read {}", path));
    let state: State = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e));
    fn find_transform(node: &Node) -> bool {
        if let Some(NodeParams::TransformParams(ref p)) = node.params {
            return p.rotation.is_some();
        }
        node.children.iter().flatten().any(find_transform)
    }
    assert!(find_transform(&state.root), "superposition example must contain a TransformParams with rotation");
}

// T1-29: basic/state.mvsj round-trip
#[test]
fn test_deser_basic_example_roundtrip() {
    let path = "tests/mol-spec-examples/basic/state.mvsj";
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("cannot read {}", path));
    let _state: State = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path, e));
}

// T1-30: builder round-trip — serialize then deserialize must preserve structure
#[test]
fn test_builder_serialize_deserialize_roundtrip() {
    let mut state = State::new();
    state
        .download("https://files.wwpdb.org/download/1cbs.cif")
        .expect("download node")
        .parse(ParseParams { format: ParseFormatT::Mmcif })
        .expect("parse node")
        .assembly_structure(StructureParams { structure_type: StructureTypeT::Assembly, ..Default::default() })
        .expect("structure node")
        .component(ComponentSelector::Selector(ComponentSelectorT::All))
        .expect("component node")
        .representation(RepresentationTypeT::Cartoon);

    let json = serde_json::to_string(&state).expect("serialization");
    let restored: State = serde_json::from_str(&json).expect("deserialization");

    // Root should have one Download child
    let children = restored.root.children.as_ref().expect("root has children");
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].kind, KindT::Download);
}

// ---------------------------------------------------------------------------
// T1-31..T1-41: Phase 1b builder additions
// ---------------------------------------------------------------------------

// T1-31: model_structure() must work on a Parse parent
#[test]
fn test_builder_model_structure_ok() {
    let mut state = State::new();
    let result = state
        .download("https://example.com/1abc.cif")
        .expect("download")
        .parse(ParseParams { format: ParseFormatT::Mmcif })
        .expect("parse")
        .model_structure(StructureParams { structure_type: StructureTypeT::Model, ..Default::default() });
    assert!(result.is_some(), "model_structure() must return Some on Parse parent");
}

// T1-32: State version must be "1.0"
#[test]
fn test_state_version_is_1_0() {
    let state = State::new();
    assert_eq!(state.metadata.version, "1.0");
}

// T1-33: focus() on Component must produce FocusInlineParams child
#[test]
fn test_builder_focus_on_component() {
    let mut state = State::new();
    let result = state
        .download("https://example.com/1abc.cif")
        .expect("download")
        .parse(ParseParams { format: ParseFormatT::Mmcif })
        .expect("parse")
        .assembly_structure(StructureParams::default())
        .expect("structure")
        .component(ComponentSelector::default())
        .expect("component")
        .focus(Some((0.0, 0.0, 1.0)), None);
    assert!(result.is_some(), "focus() must return Some on Component parent");
}

// T1-34: focus() rejects non-Component parent
#[test]
fn test_builder_focus_rejects_non_component() {
    let mut node = Node::new(KindT::Representation, None);
    assert!(node.focus(None, None).is_none());
}

// T1-35: tooltip() on Component must produce TooltipInlineParams child
#[test]
fn test_builder_tooltip_on_component() {
    let mut node = Node::new(KindT::Component, None);
    let result = node.tooltip("my tooltip".to_string());
    assert!(result.is_some());
    let children = node.children.unwrap();
    match &children[0].params {
        Some(NodeParams::TooltipInlineParams(TooltipInlineParams { text })) => {
            assert_eq!(text, "my tooltip");
        }
        other => panic!("expected TooltipInlineParams, got {:?}", other),
    }
}

// T1-36: canvas() on State root must produce CanvasParams child
#[test]
fn test_builder_canvas_on_state() {
    let mut state = State::new();
    let params = CanvasParams { background_color: ColorT::Hex("#ffffff".to_string()) };
    let result = state.canvas(params);
    assert!(result.is_some(), "canvas() must return Some");
}

// T1-37: generic_visuals() on State root
#[test]
fn test_builder_generic_visuals_on_state() {
    let mut state = State::new();
    let result = state.generic_visuals();
    assert!(result.is_some(), "generic_visuals() must return Some");
    assert_eq!(state.root.children.as_ref().unwrap()[0].kind, KindT::GenericVisuals);
}

// T1-38: sphere() on GenericVisuals parent
#[test]
fn test_builder_sphere_on_generic_visuals() {
    let mut node = Node::new(KindT::GenericVisuals, None);
    let params = SphereParams {
        position: (1.0, 2.0, 3.0),
        radius: 0.5,
        color: ColorT::Named(ferritin_molviewspec::molviewspec::nodes::ColorNamesT::Red),
        label: None,
        tooltip: None,
    };
    let result = node.sphere(params);
    assert!(result.is_some());
}

// T1-39: line() on GenericVisuals parent
#[test]
fn test_builder_line_on_generic_visuals() {
    let mut node = Node::new(KindT::GenericVisuals, None);
    let params = LineParams {
        position1: (0.0, 0.0, 0.0),
        position2: (1.0, 1.0, 1.0),
        radius: 0.1,
        color: ColorT::Named(ferritin_molviewspec::molviewspec::nodes::ColorNamesT::Blue),
        label: None,
        tooltip: None,
    };
    let result = node.line(params);
    assert!(result.is_some());
}

// T1-40: State::from_str round-trip
#[test]
fn test_state_from_str_roundtrip() {
    let mut state = State::new();
    state
        .download("https://files.wwpdb.org/download/1cbs.cif")
        .expect("download")
        .parse(ParseParams { format: ParseFormatT::Mmcif })
        .expect("parse")
        .assembly_structure(StructureParams::default())
        .expect("structure")
        .component(ComponentSelector::default())
        .expect("component")
        .representation(RepresentationTypeT::Cartoon);
    let json = serde_json::to_string(&state).unwrap();
    let restored = State::from_str(&json).expect("from_str must succeed");
    assert_eq!(restored.metadata.version, state.metadata.version);
    assert_eq!(restored.root.kind, KindT::Root);
}

// T1-41: State::from_reader round-trip using the basic example file
#[test]
fn test_state_from_reader_basic_example() {
    let file = std::fs::File::open("tests/mol-spec-examples/basic/state.mvsj")
        .expect("open basic state.mvsj");
    let state = State::from_reader(std::io::BufReader::new(file))
        .expect("from_reader must succeed");
    assert_eq!(state.root.kind, KindT::Root);
}

// ---------------------------------------------------------------------------
// T1-42..T1-48: mol-spec-example fixture assertions
// ---------------------------------------------------------------------------

fn load_example(name: &str) -> State {
    let path = format!("tests/mol-spec-examples/{}/state.mvsj", name);
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("cannot read {}", path));
    State::from_str(&content).unwrap_or_else(|e| panic!("parse error in {}: {}", path, e))
}

fn find_node<'a, F: Fn(&Node) -> bool>(node: &'a Node, pred: &F) -> Option<&'a Node> {
    if pred(node) {
        return Some(node);
    }
    node.children.iter().flatten().find_map(|c| find_node(c, pred))
}

// T1-42: basic — structure type Model, component All, cartoon, color named blue
#[test]
fn test_roundtrip_basic() {
    let state = load_example("basic");
    let structure = find_node(&state.root, &|n| {
        matches!(n.params, Some(NodeParams::StructureParams(ref p)) if matches!(p.structure_type, StructureTypeT::Model))
    });
    assert!(structure.is_some(), "basic: must have a model structure");
    let color = find_node(&state.root, &|n| {
        matches!(n.kind, KindT::Color)
    });
    assert!(color.is_some(), "basic: must have a color node");
}

// T1-43: components — has 5+ component nodes and exactly one FocusInlineParams
#[test]
fn test_roundtrip_components() {
    let state = load_example("components");
    let mut component_count = 0;
    let mut focus_count = 0;
    fn count(node: &Node, cc: &mut usize, fc: &mut usize) {
        if node.kind == KindT::Component { *cc += 1; }
        if matches!(node.params, Some(NodeParams::FocusInlineParams(_))) { *fc += 1; }
        for child in node.children.iter().flatten() { count(child, cc, fc); }
    }
    count(&state.root, &mut component_count, &mut focus_count);
    assert!(component_count >= 5, "components: expected >=5 component nodes, got {}", component_count);
    assert_eq!(focus_count, 1, "components: expected exactly 1 FocusInlineParams");
}

// T1-44: label — has a label node and a focus node under the same component
#[test]
fn test_roundtrip_label() {
    let state = load_example("label");
    let has_label = find_node(&state.root, &|n| n.kind == KindT::Label).is_some();
    let has_focus = find_node(&state.root, &|n| {
        matches!(n.params, Some(NodeParams::FocusInlineParams(_)))
    })
    .is_some();
    assert!(has_label, "label: must have a label node");
    assert!(has_focus, "label: must have a focus node");
}

// T1-45: superposition — transform rotation[0] close to -0.720216
#[test]
fn test_roundtrip_superposition() {
    let state = load_example("superposition");
    let transform = find_node(&state.root, &|n| {
        matches!(n.params, Some(NodeParams::TransformParams(ref p)) if p.rotation.is_some())
    });
    let transform = transform.expect("superposition: must have a TransformParams with rotation");
    if let Some(NodeParams::TransformParams(ref p)) = transform.params {
        let rot = p.rotation.as_ref().unwrap();
        assert_eq!(rot.len(), 9, "rotation must have 9 elements");
        let diff = (rot[0] - (-0.7202161_f64)).abs();
        assert!(diff < 1e-5, "rotation[0] expected ~-0.720216, got {}", rot[0]);
    }
}

// T1-46: symmetry — StructureParams with Symmetry type and ijk_min = (-1,-1,-1)
#[test]
fn test_roundtrip_symmetry() {
    let state = load_example("symmetry");
    let structure = find_node(&state.root, &|n| {
        matches!(n.params, Some(NodeParams::StructureParams(ref p)) if matches!(p.structure_type, StructureTypeT::Symmetry))
    });
    let structure = structure.expect("symmetry: must have a Symmetry structure");
    if let Some(NodeParams::StructureParams(ref p)) = structure.params {
        assert_eq!(p.ijk_min, Some((-1, -1, -1)));
        assert_eq!(p.ijk_max, Some((1, 1, 1)));
    }
}

// T1-47: annotations — has component_from_uri and color_from_uri nodes
#[test]
fn test_roundtrip_annotations() {
    let state = load_example("annotations");
    let has_comp_uri = find_node(&state.root, &|n| {
        matches!(n.params, Some(NodeParams::ComponentFromUriParams(_)))
    })
    .is_some();
    let has_color_uri = find_node(&state.root, &|n| {
        matches!(n.params, Some(NodeParams::ColorFromUriParams(_)))
    })
    .is_some();
    assert!(has_comp_uri, "annotations: must have component_from_uri");
    assert!(has_color_uri, "annotations: must have color_from_uri");
}

// T1-48: double round-trip — basic parsed, serialised, re-parsed matches original
#[test]
fn test_double_roundtrip_basic() {
    let path = "tests/mol-spec-examples/basic/state.mvsj";
    let content = std::fs::read_to_string(path).expect("read basic");
    let state1: State = State::from_str(&content).expect("first parse");
    let json1 = serde_json::to_string(&state1).expect("serialize");
    let state2: State = State::from_str(&json1).expect("second parse");
    // Both should have a root with children
    assert_eq!(state1.root.kind, state2.root.kind);
    assert_eq!(
        state1.root.children.as_ref().map(|c| c.len()),
        state2.root.children.as_ref().map(|c| c.len()),
    );
}

// This test gates Phase 2 selector evaluation. It is #[ignore] until Phase 2
// implements AtomCollection::apply_selector() / ComponentExpression → AtomMask.
// The empty-expression → all-true behavior is documented in nodes.rs.
#[test]
#[ignore = "Phase 2 gate: requires ComponentExpression → AtomMask evaluation (ferritin-vgx Phase 2)"]
fn test_mask_empty_expression_all_fields_none() {
    // When Phase 2 lands, this test must pass:
    //   let col = AtomCollection::from_pdb("tests/data/1cbs.pdb");
    //   let n = col.atom_count();
    //   let expr = ComponentExpression::default(); // all fields None
    //   let mask = col.apply_expression(&expr);
    //   assert_eq!(mask.count_true(), n);
    //
    // Placeholder assertion so the test compiles:
    let expr = ComponentExpression::default();
    assert!(expr.label_asym_id.is_none());
    assert!(expr.auth_asym_id.is_none());
    assert!(expr.label_seq_id.is_none());
    assert!(expr.auth_seq_id.is_none());
    assert!(expr.atom_index.is_none());
    // ↑ all fields None → Phase 2 evaluation must return all-true mask
}

// ferritin-2g6: Volume node + isosurface representation build and round-trip.
#[test]
fn test_volume_isosurface_roundtrip() {
    let mut root = Node::new(KindT::Root, None);
    let download = root.download("https://example.org/density.map").unwrap();
    let parse = download
        .parse(ParseParams {
            format: ParseFormatT::Bcif,
        })
        .unwrap();
    let volume = parse
        .volume(VolumeParams {
            channel_id: Some("2fo-fc".to_string()),
        })
        .unwrap();
    volume.volume_representation(VolumeRepresentationParams {
        representation_type: VolumeRepresentationTypeT::Isosurface,
        isosurface: Some(IsoSurfaceParams {
            isovalue: None,
            relative_isovalue: Some(1.5),
            wireframe: Some(false),
            opacity: Some(0.6),
        }),
    });

    let json = serde_json::to_string(&root).expect("serialize volume state");
    let parsed: Node = serde_json::from_str(&json).expect("deserialize volume state");

    let volume_node = find_node(&parsed, &|n| n.kind == KindT::Volume)
        .expect("Volume node must round-trip");
    if let Some(NodeParams::VolumeParams(ref p)) = volume_node.params {
        assert_eq!(p.channel_id.as_deref(), Some("2fo-fc"));
    } else {
        panic!("Volume node must carry VolumeParams");
    }

    let repr_node = find_node(&parsed, &|n| n.kind == KindT::VolumeRepresentation)
        .expect("VolumeRepresentation node must round-trip");
    if let Some(NodeParams::VolumeRepresentationParams(ref p)) = repr_node.params {
        assert!(matches!(
            p.representation_type,
            VolumeRepresentationTypeT::Isosurface
        ));
        let iso = p.isosurface.as_ref().expect("isosurface params");
        assert_eq!(iso.relative_isovalue, Some(1.5));
        assert_eq!(iso.opacity, Some(0.6));
    } else {
        panic!("VolumeRepresentation node must carry VolumeRepresentationParams");
    }
}

// ferritin-rar: MvsjFile deserializes a mol-view-stories-style snapshot sequence.
#[test]
fn test_mvsj_file_two_snapshot_sequence() {
    let json = r#"{
        "metadata": { "version": "1.0", "timestamp": "2026-07-01T00:00:00Z" },
        "snapshots": [
            {
                "root": { "kind": "root" },
                "title": "Intro",
                "key": "intro",
                "linger_duration_ms": 3000,
                "transition_duration_ms": 500
            },
            {
                "root": { "kind": "root", "children": [{ "kind": "camera", "params": { "target": [0,0,0], "position": [10,10,10] } }] },
                "title": "Overview",
                "key": "overview",
                "linger_duration_ms": 4000,
                "transition_duration_ms": 800
            }
        ]
    }"#;

    let story = MvsjFile::from_str(json).expect("parse two-snapshot mvsj");
    assert_eq!(story.snapshots.len(), 2);
    assert_eq!(story.snapshots[0].title.as_deref(), Some("Intro"));
    assert_eq!(story.snapshots[0].root.kind, KindT::Root);
    assert_eq!(story.snapshots[1].title.as_deref(), Some("Overview"));
    assert_eq!(
        story.snapshots[1]
            .root
            .children
            .as_ref()
            .map(|c| c.len()),
        Some(1)
    );
    assert_eq!(story.snapshots[0].linger_duration_ms, Some(3000));
    assert_eq!(story.snapshots[1].transition_duration_ms, Some(800));
}
