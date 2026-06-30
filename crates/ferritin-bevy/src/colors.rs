//! Color schemes for protein structure rendering.

use crate::selection::{AtomMask, evaluate_selector};
use bevy::color::Srgba;
use bevy::math::Vec4;
use bevy::prelude::{Color, Mesh};
use ferritin_core::AtomCollection;
use ferritin_molviewspec::molviewspec::nodes::{ColorNamesT, ColorT, ComponentSelector};

/// Represents different color schemes for rendering atoms.
#[derive(Clone)]
pub enum ColorScheme {
    /// A solid, single color for all atoms.
    Solid(Color),
    /// Colors atoms based on their element type (CPK scheme).
    ByAtomType,
    /// Color from a MolViewSpec `ColorT` (named or hex).
    MvsColor(ColorT),
}

impl ColorScheme {
    pub fn get_color(&self, element: &ferritin_core::info::elements::Element) -> Color {
        match self {
            ColorScheme::Solid(color) => *color,
            ColorScheme::ByAtomType => Color::Srgba(element_color(element)),
            ColorScheme::MvsColor(c) => color_t_to_bevy(c),
        }
    }
}

/// Discriminates whether a vertex_map contains atom indices or residue iteration indices.
pub enum VertexMapKind {
    /// Each entry is a 0-based atom index into the `AtomCollection`.
    AtomMap,
    /// Each entry is a 0-based amino-acid residue iteration index.
    /// Coloring is approximate: a residue is colored if any of its atoms passes the selector.
    ResidueMap,
}

/// Paint vertex colors on `mesh` by applying MVS color nodes in order.
///
/// Each `(ComponentSelector, ColorT)` pair selects atoms and assigns a color.
/// Later nodes override earlier nodes on overlapping vertices.
///
/// `vertex_map[vertex_idx]` holds the atom index (for `VertexMapKind::AtomMap`) or the
/// amino-acid residue iteration index (for `VertexMapKind::ResidueMap`) that produced
/// that vertex.
pub fn apply_mvs_colors(
    mesh: &mut Mesh,
    color_nodes: &[(ComponentSelector, ColorT)],
    model: &AtomCollection,
    vertex_map: &[usize],
    map_kind: VertexMapKind,
) {
    let n_verts = vertex_map.len();
    if n_verts == 0 {
        return;
    }

    // Default: white for all vertices.
    let mut vertex_colors: Vec<Vec4> = vec![Vec4::new(1.0, 1.0, 1.0, 1.0); n_verts];

    for (selector, color_t) in color_nodes {
        let atom_mask = evaluate_selector(selector, model);
        let bevy_color = color_t_to_bevy(color_t).to_srgba();
        let cv = Vec4::new(
            bevy_color.red,
            bevy_color.green,
            bevy_color.blue,
            bevy_color.alpha,
        );

        match map_kind {
            VertexMapKind::AtomMap => {
                for (v, &atom_idx) in vertex_map.iter().enumerate() {
                    if atom_mask.0.get(atom_idx).copied().unwrap_or(false) {
                        vertex_colors[v] = cv;
                    }
                }
            }
            VertexMapKind::ResidueMap => {
                // Build a per-(residue-iter-idx) boolean: true if any atom in that residue passes.
                let residue_passes = residue_mask_from_atom_mask(model, &atom_mask);
                for (v, &res_iter_idx) in vertex_map.iter().enumerate() {
                    if residue_passes.get(res_iter_idx).copied().unwrap_or(false) {
                        vertex_colors[v] = cv;
                    }
                }
            }
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);
}

/// Build a per-residue boolean mask (indexed by ALL-residue iteration order)
/// from a per-atom mask: residue passes if ANY of its atoms passes.
fn residue_mask_from_atom_mask(model: &AtomCollection, atom_mask: &AtomMask) -> Vec<bool> {
    model
        .iter_residues()
        .map(|residue| {
            residue
                .atom_range()
                .any(|i| atom_mask.0.get(i).copied().unwrap_or(false))
        })
        .collect()
}

/// Convert a MolViewSpec `ColorT` to a Bevy `Color`.
pub fn color_t_to_bevy(color: &ColorT) -> Color {
    match color {
        ColorT::Hex(s) => parse_hex_color(s),
        ColorT::Named(name) => named_color_to_bevy(name),
    }
}

/// Parse `#rrggbb` (case-insensitive, leading `#` optional).
/// Returns `Color::WHITE` with a `warn!` on any parse error.
pub fn parse_hex_color(s: &str) -> Color {
    let s = s.trim_start_matches('#');
    if s.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return Color::srgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
        }
    }
    bevy::log::warn!("Invalid hex color '{}'; falling back to white", s);
    Color::WHITE
}

fn named_color_to_bevy(name: &ColorNamesT) -> Color {
    let (r, g, b) = css_named_color(name);
    Color::srgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

/// CSS named color RGB values (W3C spec, https://www.w3.org/TR/css-color-3/).
#[rustfmt::skip]
pub fn css_named_color(name: &ColorNamesT) -> (u8, u8, u8) {
    match name {
        ColorNamesT::Aliceblue           => (240, 248, 255),
        ColorNamesT::Antiquewhite        => (250, 235, 215),
        ColorNamesT::Aqua                => (0,   255, 255),
        ColorNamesT::Aquamarine          => (127, 255, 212),
        ColorNamesT::Azure               => (240, 255, 255),
        ColorNamesT::Beige               => (245, 245, 220),
        ColorNamesT::Bisque              => (255, 228, 196),
        ColorNamesT::Black               => (0,   0,   0  ),
        ColorNamesT::Blanchedalmond      => (255, 235, 205),
        ColorNamesT::Blue                => (0,   0,   255),
        ColorNamesT::Blueviolet          => (138, 43,  226),
        ColorNamesT::Brown               => (165, 42,  42 ),
        ColorNamesT::Burlywood           => (222, 184, 135),
        ColorNamesT::Cadetblue           => (95,  158, 160),
        ColorNamesT::Chartreuse          => (127, 255, 0  ),
        ColorNamesT::Chocolate           => (210, 105, 30 ),
        ColorNamesT::Coral               => (255, 127, 80 ),
        ColorNamesT::Cornflowerblue      => (100, 149, 237),
        ColorNamesT::Cornsilk            => (255, 248, 220),
        ColorNamesT::Crimson             => (220, 20,  60 ),
        ColorNamesT::Cyan                => (0,   255, 255),
        ColorNamesT::Darkblue            => (0,   0,   139),
        ColorNamesT::Darkcyan            => (0,   139, 139),
        ColorNamesT::Darkgoldenrod       => (184, 134, 11 ),
        ColorNamesT::Darkgray            => (169, 169, 169),
        ColorNamesT::Darkgreen           => (0,   100, 0  ),
        ColorNamesT::Darkgrey            => (169, 169, 169),
        ColorNamesT::Darkkhaki           => (189, 183, 107),
        ColorNamesT::Darkmagenta         => (139, 0,   139),
        ColorNamesT::Darkolivegreen      => (85,  107, 47 ),
        ColorNamesT::Darkorange          => (255, 140, 0  ),
        ColorNamesT::Darkorchid          => (153, 50,  204),
        ColorNamesT::Darkred             => (139, 0,   0  ),
        ColorNamesT::Darksalmon          => (233, 150, 122),
        ColorNamesT::Darkseagreen        => (143, 188, 143),
        ColorNamesT::Darkslateblue       => (72,  61,  139),
        ColorNamesT::Darkslategray       => (47,  79,  79 ),
        ColorNamesT::Darkslategrey       => (47,  79,  79 ),
        ColorNamesT::Darkturquoise       => (0,   206, 209),
        ColorNamesT::Darkviolet          => (148, 0,   211),
        ColorNamesT::Deeppink            => (255, 20,  147),
        ColorNamesT::Deepskyblue         => (0,   191, 255),
        ColorNamesT::Dimgray             => (105, 105, 105),
        ColorNamesT::Dimgrey             => (105, 105, 105),
        ColorNamesT::Dodgerblue          => (30,  144, 255),
        ColorNamesT::Firebrick           => (178, 34,  34 ),
        ColorNamesT::Floralwhite         => (255, 250, 240),
        ColorNamesT::Forestgreen         => (34,  139, 34 ),
        ColorNamesT::Fuchsia             => (255, 0,   255),
        ColorNamesT::Gainsboro           => (220, 220, 220),
        ColorNamesT::Ghostwhite          => (248, 248, 255),
        ColorNamesT::Gold                => (255, 215, 0  ),
        ColorNamesT::Goldenrod           => (218, 165, 32 ),
        ColorNamesT::Gray                => (128, 128, 128),
        ColorNamesT::Green               => (0,   128, 0  ),
        ColorNamesT::Greenyellow         => (173, 255, 47 ),
        ColorNamesT::Grey                => (128, 128, 128),
        ColorNamesT::Honeydew            => (240, 255, 240),
        ColorNamesT::Hotpink             => (255, 105, 180),
        ColorNamesT::Indianred           => (205, 92,  92 ),
        ColorNamesT::Indigo              => (75,  0,   130),
        ColorNamesT::Ivory               => (255, 255, 240),
        ColorNamesT::Khaki               => (240, 230, 140),
        ColorNamesT::Lavender            => (230, 230, 250),
        ColorNamesT::Lavenderblush       => (255, 240, 245),
        ColorNamesT::Lawngreen           => (124, 252, 0  ),
        ColorNamesT::Lemonchiffon        => (255, 250, 205),
        ColorNamesT::Lightblue           => (173, 216, 230),
        ColorNamesT::Lightcoral          => (240, 128, 128),
        ColorNamesT::Lightcyan           => (224, 255, 255),
        ColorNamesT::Lightgoldenrodyellow=> (250, 250, 210),
        ColorNamesT::Lightgray           => (211, 211, 211),
        ColorNamesT::Lightgreen          => (144, 238, 144),
        ColorNamesT::Lightgrey           => (211, 211, 211),
        ColorNamesT::Lightpink           => (255, 182, 193),
        ColorNamesT::Lightsalmon         => (255, 160, 122),
        ColorNamesT::Lightseagreen       => (32,  178, 170),
        ColorNamesT::Lightskyblue        => (135, 206, 250),
        ColorNamesT::Lightslategray      => (119, 136, 153),
        ColorNamesT::Lightslategrey      => (119, 136, 153),
        ColorNamesT::Lightsteelblue      => (176, 196, 222),
        ColorNamesT::Lightyellow         => (255, 255, 224),
        ColorNamesT::Lime                => (0,   255, 0  ),
        ColorNamesT::Limegreen           => (50,  205, 50 ),
        ColorNamesT::Linen               => (250, 240, 230),
        ColorNamesT::Magenta             => (255, 0,   255),
        ColorNamesT::Maroon              => (128, 0,   0  ),
        ColorNamesT::Mediumaquamarine    => (102, 205, 170),
        ColorNamesT::Mediumblue          => (0,   0,   205),
        ColorNamesT::Mediumorchid        => (186, 85,  211),
        ColorNamesT::Mediumpurple        => (147, 112, 219),
        ColorNamesT::Mediumseagreen      => (60,  179, 113),
        ColorNamesT::Mediumslateblue     => (123, 104, 238),
        ColorNamesT::Mediumspringgreen   => (0,   250, 154),
        ColorNamesT::Mediumturquoise     => (72,  209, 204),
        ColorNamesT::Mediumvioletred     => (199, 21,  133),
        ColorNamesT::Midnightblue        => (25,  25,  112),
        ColorNamesT::Mintcream           => (245, 255, 250),
        ColorNamesT::Mistyrose           => (255, 228, 225),
        ColorNamesT::Moccasin            => (255, 228, 181),
        ColorNamesT::Navajowhite         => (255, 222, 173),
        ColorNamesT::Navy                => (0,   0,   128),
        ColorNamesT::Oldlace             => (253, 245, 230),
        ColorNamesT::Olive               => (128, 128, 0  ),
        ColorNamesT::Olivedrab           => (107, 142, 35 ),
        ColorNamesT::Orange              => (255, 165, 0  ),
        ColorNamesT::Orangered           => (255, 69,  0  ),
        ColorNamesT::Orchid              => (218, 112, 214),
        ColorNamesT::Palegoldenrod       => (238, 232, 170),
        ColorNamesT::Palegreen           => (152, 251, 152),
        ColorNamesT::Paleturquoise       => (175, 238, 238),
        ColorNamesT::Palevioletred       => (219, 112, 147),
        ColorNamesT::Papayawhip          => (255, 239, 213),
        ColorNamesT::Peachpuff           => (255, 218, 185),
        ColorNamesT::Peru                => (205, 133, 63 ),
        ColorNamesT::Pink                => (255, 192, 203),
        ColorNamesT::Plum                => (221, 160, 221),
        ColorNamesT::Powderblue          => (176, 224, 230),
        ColorNamesT::Purple              => (128, 0,   128),
        ColorNamesT::Red                 => (255, 0,   0  ),
        ColorNamesT::Rosybrown           => (188, 143, 143),
        ColorNamesT::Royalblue           => (65,  105, 225),
        ColorNamesT::Saddlebrown         => (139, 69,  19 ),
        ColorNamesT::Salmon              => (250, 128, 114),
        ColorNamesT::Sandybrown          => (244, 164, 96 ),
        ColorNamesT::Seagreen            => (46,  139, 87 ),
        ColorNamesT::Seashell            => (255, 245, 238),
        ColorNamesT::Sienna              => (160, 82,  45 ),
        ColorNamesT::Silver              => (192, 192, 192),
        ColorNamesT::Skyblue             => (135, 206, 235),
        ColorNamesT::Slateblue           => (106, 90,  205),
        ColorNamesT::Slategray           => (112, 128, 144),
        ColorNamesT::Slategrey           => (112, 128, 144),
        ColorNamesT::Snow                => (255, 250, 250),
        ColorNamesT::Springgreen         => (0,   255, 127),
        ColorNamesT::Steelblue           => (70,  130, 180),
        ColorNamesT::Tan                 => (210, 180, 140),
        ColorNamesT::Teal                => (0,   128, 128),
        ColorNamesT::Thistle             => (216, 191, 216),
        ColorNamesT::Tomato              => (255, 99,  71 ),
        ColorNamesT::Turquoise           => (64,  224, 208),
        ColorNamesT::Violet              => (238, 130, 238),
        ColorNamesT::Wheat               => (245, 222, 179),
        ColorNamesT::White               => (255, 255, 255),
        ColorNamesT::Whitesmoke          => (245, 245, 245),
        ColorNamesT::Yellow              => (255, 255, 0  ),
        ColorNamesT::Yellowgreen         => (154, 205, 50 ),
    }
}

/// CPK element colors.
#[rustfmt::skip]
pub fn element_color(element: &ferritin_core::info::elements::Element) -> Srgba {
    use ferritin_core::info::elements::Element;
    match element {
        Element::H  => Srgba::rgb(1.0, 1.0, 1.0),
        Element::C  => Srgba::rgb(0.5, 0.5, 0.5),
        Element::N  => Srgba::rgb(0.0, 0.0, 1.0),
        Element::O  => Srgba::rgb(1.0, 0.0, 0.0),
        Element::P  => Srgba::rgb(1.0, 0.5, 0.0),
        Element::S  => Srgba::rgb(1.0, 1.0, 0.0),
        Element::Cl => Srgba::rgb(0.0, 1.0, 0.0),
        Element::Fe => Srgba::rgb(0.6, 0.0, 0.0),
        Element::Ca => Srgba::rgb(0.5, 0.5, 0.5),
        Element::Mg => Srgba::rgb(0.5, 1.0, 0.0),
        Element::Na => Srgba::rgb(0.0, 0.0, 1.0),
        Element::K  => Srgba::rgb(0.8, 0.6, 1.0),
        Element::Zn => Srgba::rgb(0.6, 0.6, 0.6),
        Element::Cu => Srgba::rgb(0.8, 0.4, 0.0),
        Element::F  => Srgba::rgb(0.7, 1.0, 1.0),
        Element::Br => Srgba::rgb(0.6, 0.1, 0.1),
        Element::I  => Srgba::rgb(0.4, 0.0, 0.7),
        Element::B  => Srgba::rgb(1.0, 0.7, 0.7),
        Element::Se => Srgba::rgb(1.0, 0.5, 0.0),
        _           => Srgba::rgb(0.5, 0.5, 0.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::prelude::Mesh;
    use bevy::render::mesh::{PrimitiveTopology, VertexAttributeValues};
    use ferritin_core::AtomCollection;
    use ferritin_core::info::elements::Element;
    use ferritin_molviewspec::molviewspec::nodes::{
        ColorNamesT, ColorT, ComponentExpression, ComponentSelector, ComponentSelectorT,
    };

    /// 9-atom mixed collection for selector/color tests.
    /// Atoms: ALA(3, A/1), GLY(2, A/2), HOH(1, B/3 hetero), ZN(1, B/4 hetero), LIG(2, B/5 hetero)
    fn make_test_collection() -> AtomCollection {
        let coords = vec![[0.0f32; 3]; 9];
        let res_ids: Vec<i32> = vec![1, 1, 1, 2, 2, 3, 4, 5, 5];
        let res_names: Vec<String> = ["ALA", "ALA", "ALA", "GLY", "GLY", "HOH", "ZN", "LIG", "LIG"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let is_hetero = vec![false, false, false, false, false, true, true, true, true];
        let elements = vec![
            Element::N,
            Element::C,
            Element::C,
            Element::N,
            Element::C,
            Element::O,
            Element::Zn,
            Element::C,
            Element::O,
        ];
        let atom_names: Vec<String> = ["N", "CA", "C", "N", "CA", "O", "ZN", "C1", "O1"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let chain_ids: Vec<String> = ["A", "A", "A", "A", "A", "B", "B", "B", "B"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        AtomCollection::new(
            9, coords, res_ids, res_names, is_hetero, elements, atom_names, chain_ids, None,
        )
    }

    /// Build a minimal Mesh with `n_verts` vertices and a vertex_map [0, 0, 1, 1, ..].
    fn make_dummy_mesh_and_map(n_verts: usize, n_atoms: usize) -> (Mesh, Vec<usize>) {
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        // 2 vertices per atom, cycling through atom indices
        let map = (0..n_verts).map(|v| (v / 2).min(n_atoms - 1)).collect();
        (mesh, map)
    }

    fn get_colors(mesh: &Mesh) -> Option<Vec<[f32; 4]>> {
        match mesh.attribute(Mesh::ATTRIBUTE_COLOR)? {
            VertexAttributeValues::Float32x4(c) => Some(c.clone()),
            _ => None,
        }
    }

    // T2-41: Single color node (All, hex) → all vertices get that color
    #[test]
    fn test_apply_single_color_all() {
        let ac = make_test_collection();
        let (mut mesh, map) = make_dummy_mesh_and_map(18, 9);
        let color_nodes = vec![(
            ComponentSelector::Selector(ComponentSelectorT::All),
            ColorT::Hex("#4b7fcc".to_string()),
        )];
        apply_mvs_colors(&mut mesh, &color_nodes, &ac, &map, VertexMapKind::AtomMap);
        let colors = get_colors(&mesh).expect("COLOR attribute must be set");
        assert_eq!(colors.len(), 18);
        let expected_r = 0x4b as f32 / 255.0;
        let expected_g = 0x7f as f32 / 255.0;
        let expected_b = 0xcc as f32 / 255.0;
        for c in &colors {
            assert!((c[0] - expected_r).abs() < 1e-4, "red mismatch");
            assert!((c[1] - expected_g).abs() < 1e-4, "green mismatch");
            assert!((c[2] - expected_b).abs() < 1e-4, "blue mismatch");
        }
    }

    // T2-42: Two color nodes; later (expression) overrides earlier (All)
    #[test]
    fn test_apply_two_colors_base_then_override() {
        let ac = make_test_collection();
        // 2 verts per atom, 9 atoms = 18 verts; map: v/2 → atom_idx
        let (mut mesh, map) = make_dummy_mesh_and_map(18, 9);
        // ALA residue = atoms 0,1,2 → verts 0..5
        let color_nodes = vec![
            (
                ComponentSelector::Selector(ComponentSelectorT::All),
                ColorT::Named(ColorNamesT::Blue),
            ),
            (
                ComponentSelector::Expression(ComponentExpression {
                    auth_seq_id: Some(1), // ALA residue (atoms 0,1,2)
                    ..Default::default()
                }),
                ColorT::Named(ColorNamesT::Red),
            ),
        ];
        apply_mvs_colors(&mut mesh, &color_nodes, &ac, &map, VertexMapKind::AtomMap);
        let colors = get_colors(&mesh).unwrap();
        // First 6 verts (atoms 0,1,2 = ALA/res 1) should be red
        for c in &colors[0..6] {
            assert!(c[0] > 0.99, "expected red for ALA res 1 verts");
            assert!(c[2] < 0.01, "expected no blue for ALA res 1 verts");
        }
        // Remaining verts should be blue
        for c in &colors[6..] {
            assert!(c[2] > 0.99, "expected blue for non-res-1 verts");
            assert!(c[0] < 0.01, "expected no red for non-res-1 verts");
        }
    }

    // T2-43: Order matters — (expression, red) then (All, blue) → all blue
    #[test]
    fn test_apply_colors_order_matters_reversed() {
        let ac = make_test_collection();
        let (mut mesh, map) = make_dummy_mesh_and_map(18, 9);
        let color_nodes = vec![
            (
                ComponentSelector::Expression(ComponentExpression {
                    auth_seq_id: Some(1),
                    ..Default::default()
                }),
                ColorT::Named(ColorNamesT::Red),
            ),
            (
                ComponentSelector::Selector(ComponentSelectorT::All),
                ColorT::Named(ColorNamesT::Blue),
            ),
        ];
        apply_mvs_colors(&mut mesh, &color_nodes, &ac, &map, VertexMapKind::AtomMap);
        let colors = get_colors(&mesh).unwrap();
        for (i, c) in colors.iter().enumerate() {
            assert!(
                c[2] > 0.99,
                "vert {i}: expected all-blue after All override"
            );
            assert!(c[0] < 0.01, "vert {i}: expected no red");
        }
    }

    // T2-44: Middle non-matching selector is a no-op; last All wins
    #[test]
    fn test_apply_colors_middle_no_match() {
        let ac = make_test_collection();
        let (mut mesh, map) = make_dummy_mesh_and_map(18, 9);
        let color_nodes = vec![
            (
                ComponentSelector::Selector(ComponentSelectorT::All),
                ColorT::Named(ColorNamesT::Blue),
            ),
            (
                ComponentSelector::Expression(ComponentExpression {
                    auth_asym_id: Some("Z".to_string()), // nonexistent chain
                    ..Default::default()
                }),
                ColorT::Named(ColorNamesT::Red),
            ),
            (
                ComponentSelector::Selector(ComponentSelectorT::All),
                ColorT::Named(ColorNamesT::Green),
            ),
        ];
        apply_mvs_colors(&mut mesh, &color_nodes, &ac, &map, VertexMapKind::AtomMap);
        let colors = get_colors(&mesh).unwrap();
        for (i, c) in colors.iter().enumerate() {
            // Green = (0, 128, 0) / 255 ≈ (0, 0.502, 0)
            assert!(c[0] < 0.01, "vert {i}: expected no red");
            assert!(c[1] > 0.49, "vert {i}: expected green channel");
            assert!(c[2] < 0.01, "vert {i}: expected no blue");
        }
    }

    // T2-45: No color nodes → white fallback on all vertices
    #[test]
    fn test_apply_colors_no_color_nodes() {
        let ac = make_test_collection();
        let (mut mesh, map) = make_dummy_mesh_and_map(6, 9);
        apply_mvs_colors(&mut mesh, &[], &ac, &map, VertexMapKind::AtomMap);
        let colors = get_colors(&mesh).unwrap();
        for (i, c) in colors.iter().enumerate() {
            assert!((c[0] - 1.0).abs() < 1e-4, "vert {i}: expected white (red)");
            assert!(
                (c[1] - 1.0).abs() < 1e-4,
                "vert {i}: expected white (green)"
            );
            assert!((c[2] - 1.0).abs() < 1e-4, "vert {i}: expected white (blue)");
        }
    }

    // T2-46: apply_mvs_colors never changes vertex count
    #[test]
    fn test_apply_colors_preserves_vertex_count() {
        let ac = make_test_collection();
        let (mut mesh, map) = make_dummy_mesh_and_map(18, 9);
        let before = map.len();
        let color_nodes = vec![(
            ComponentSelector::Selector(ComponentSelectorT::All),
            ColorT::Named(ColorNamesT::Blue),
        )];
        apply_mvs_colors(&mut mesh, &color_nodes, &ac, &map, VertexMapKind::AtomMap);
        let colors = get_colors(&mesh).unwrap();
        assert_eq!(colors.len(), before, "vertex count must not change");
    }

    // T2-48: hex lowercase
    #[test]
    fn test_parse_hex_lowercase() {
        let color = parse_hex_color("#1b9e77");
        let srgb = color.to_srgba();
        assert!((srgb.red - 0x1b as f32 / 255.0).abs() < 1e-4);
        assert!((srgb.green - 0x9e as f32 / 255.0).abs() < 1e-4);
        assert!((srgb.blue - 0x77 as f32 / 255.0).abs() < 1e-4);
    }

    // T2-42: hex uppercase
    #[test]
    fn test_parse_hex_uppercase() {
        let c1 = parse_hex_color("#FF0000");
        let c2 = parse_hex_color("#ff0000");
        let s1 = c1.to_srgba();
        let s2 = c2.to_srgba();
        assert!((s1.red - s2.red).abs() < 1e-6);
    }

    // T2-43: hex without leading #
    #[test]
    fn test_parse_hex_no_hash() {
        let color = parse_hex_color("0000ff");
        let srgb = color.to_srgba();
        assert!(srgb.blue > 0.99);
        assert!(srgb.red < 0.01);
    }

    // T2-44: invalid hex → white
    #[test]
    fn test_parse_hex_invalid_white_fallback() {
        let color = parse_hex_color("not-a-color");
        let srgb = color.to_srgba();
        assert!((srgb.red - 1.0).abs() < 1e-4);
        assert!((srgb.green - 1.0).abs() < 1e-4);
        assert!((srgb.blue - 1.0).abs() < 1e-4);
    }

    // T2-45: named blue
    #[test]
    fn test_named_blue() {
        let color = color_t_to_bevy(&ColorT::Named(ColorNamesT::Blue));
        let srgb = color.to_srgba();
        assert!(srgb.blue > 0.99);
        assert!(srgb.red < 0.01);
        assert!(srgb.green < 0.01);
    }

    // T2-46: named red
    #[test]
    fn test_named_red() {
        let color = color_t_to_bevy(&ColorT::Named(ColorNamesT::Red));
        let srgb = color.to_srgba();
        assert!(srgb.red > 0.99);
        assert!(srgb.blue < 0.01);
    }

    // T2-47: named black
    #[test]
    fn test_named_black() {
        let color = color_t_to_bevy(&ColorT::Named(ColorNamesT::Black));
        let srgb = color.to_srgba();
        assert!(srgb.red < 0.01);
        assert!(srgb.green < 0.01);
        assert!(srgb.blue < 0.01);
    }

    // T2-48: named white
    #[test]
    fn test_named_white() {
        let color = color_t_to_bevy(&ColorT::Named(ColorNamesT::White));
        let srgb = color.to_srgba();
        assert!((srgb.red - 1.0).abs() < 1e-4);
        assert!((srgb.green - 1.0).abs() < 1e-4);
        assert!((srgb.blue - 1.0).abs() < 1e-4);
    }

    // T2-49: ColorScheme::MvsColor round-trip
    #[test]
    fn test_color_scheme_mvscolor() {
        let scheme = ColorScheme::MvsColor(ColorT::Hex("#ff0000".to_string()));
        let color = scheme.get_color(&ferritin_core::info::elements::Element::C);
        let srgb = color.to_srgba();
        assert!(srgb.red > 0.99);
        assert!(srgb.blue < 0.01);
    }
}
