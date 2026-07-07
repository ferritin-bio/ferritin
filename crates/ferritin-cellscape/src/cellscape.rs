//! Core Cellscape Namespace
//!
//! Flattens 3D atomic structures into 2D SVG "footprint" renderings, in the
//! spirit of the Python [Cellscape](https://github.com/jordisr/cellscape) tool:
//! each residue is drawn as the union of its atoms' van der Waals circles,
//! projected onto a chosen plane and normalized to fill the output canvas.

use ferritin_core::Model;
use geo::{BooleanOps, Coord, LineString, MultiPolygon, Point, Polygon};
use std::f64::consts::PI;
use svg::Document;
use svg::node::element::Path;

/// A rotating palette of visually distinct per-chain SVG fill colors.
const CHAIN_PALETTE: &[&str] = &[
    "#4a90d9", "#d98f4a", "#4ac367", "#c34a67", "#af4ad9", "#4ac3c3", "#e0e04a", "#e04a4a",
];

/// Fallback van der Waals radius (Å) for elements with no tabulated value.
const DEFAULT_VDW_RADIUS: f64 = 1.5;

/// Which pair of Cartesian axes to project onto when flattening to 2D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionAxis {
    Xy,
    Xz,
    Yz,
}

impl ProjectionAxis {
    fn project(&self, coords: [f32; 3]) -> (f64, f64) {
        let [x, y, z] = coords.map(|c| c as f64);
        match self {
            ProjectionAxis::Xy => (x, y),
            ProjectionAxis::Xz => (x, z),
            ProjectionAxis::Yz => (y, z),
        }
    }
}

/// Rendering options for [`StructureFlatten::flatten_structure_with`].
#[derive(Clone, Debug)]
pub struct FlattenOptions {
    /// Which plane to project 3D coordinates onto.
    pub projection: ProjectionAxis,
    /// Output SVG width/height in pixels (square canvas).
    pub canvas_size: f64,
    /// Empty margin kept around the drawn structure, in canvas pixels.
    pub padding: f64,
    /// Color each residue by its chain (rotating palette) instead of solid blue.
    pub color_by_chain: bool,
}

impl Default for FlattenOptions {
    fn default() -> Self {
        Self {
            projection: ProjectionAxis::Xy,
            canvas_size: 800.0,
            padding: 20.0,
            color_by_chain: true,
        }
    }
}

/// Flattens 3D atomic structures into a 2D SVG representation.
pub trait StructureFlatten {
    /// Flatten using [`FlattenOptions::default`].
    fn flatten_structure(&self) -> Document {
        self.flatten_structure_with(&FlattenOptions::default())
    }

    /// Flatten with explicit projection/canvas/coloring options.
    fn flatten_structure_with(&self, options: &FlattenOptions) -> Document;
}

/// One atom's 2D projection and radius, prior to canvas scaling.
struct ProjectedAtom {
    chain_id: String,
    center: (f64, f64),
    radius: f64,
}

impl StructureFlatten for Model {
    fn flatten_structure_with(&self, options: &FlattenOptions) -> Document {
        // One inner Vec per amino-acid residue; empty residues (no atoms) are dropped.
        let residues: Vec<Vec<ProjectedAtom>> = self
            .residues_aminoacid()
            .map(|residue| {
                let chain_id = residue.chain_id().to_string();
                residue
                    .iter_atoms()
                    .map(|atom| {
                        let center = options.projection.project(atom.coords());
                        let radius = atom
                            .element()
                            .atomic_radius()
                            .van_der_waals
                            .unwrap_or(DEFAULT_VDW_RADIUS);
                        ProjectedAtom { chain_id: chain_id.clone(), center, radius }
                    })
                    .collect()
            })
            .filter(|atoms: &Vec<ProjectedAtom>| !atoms.is_empty())
            .collect();

        let base = Document::new()
            .set("width", options.canvas_size)
            .set("height", options.canvas_size)
            .set("viewBox", (0.0, 0.0, options.canvas_size, options.canvas_size));

        let Some((min_x, min_y, max_x, max_y)) = bounding_box(residues.iter().flatten()) else {
            return base;
        };

        // Scale+translate so the structure's bounding box (atom centers ± radius)
        // fills the canvas minus `padding` on every side, preserving aspect ratio.
        let width = (max_x - min_x).max(f64::EPSILON);
        let height = (max_y - min_y).max(f64::EPSILON);
        let drawable = (options.canvas_size - 2.0 * options.padding).max(1.0);
        let scale = drawable / width.max(height);
        let offset_x = options.padding + (drawable - width * scale) / 2.0 - min_x * scale;
        let offset_y = options.padding + (drawable - height * scale) / 2.0 - min_y * scale;

        let mut chain_order: Vec<String> = Vec::new();
        residues.iter().fold(base, |doc, atoms| {
            let circles: Vec<Polygon<f64>> = atoms
                .iter()
                .map(|a| {
                    let cx = a.center.0 * scale + offset_x;
                    let cy = a.center.1 * scale + offset_y;
                    create_circle(&Point::new(cx, cy), a.radius * scale)
                })
                .collect();

            let merged = circles.iter().skip(1).fold(
                MultiPolygon(vec![circles[0].clone()]),
                |acc, circle| acc.union(circle),
            );

            let fill = if options.color_by_chain {
                let chain = &atoms[0].chain_id;
                let idx = chain_order.iter().position(|c| c == chain).unwrap_or_else(|| {
                    chain_order.push(chain.clone());
                    chain_order.len() - 1
                });
                CHAIN_PALETTE[idx % CHAIN_PALETTE.len()]
            } else {
                "blue"
            };

            merged.0.iter().fold(doc, |doc, polygon| {
                doc.add(create_svg_path(polygon, fill, "black", 1.0))
            })
        })
    }
}

/// Axis-aligned bounding box `(min_x, min_y, max_x, max_y)` over atom circles
/// (center ± radius), or `None` if there are no atoms.
fn bounding_box<'a>(atoms: impl Iterator<Item = &'a ProjectedAtom>) -> Option<(f64, f64, f64, f64)> {
    atoms.fold(None, |acc, atom| {
        let (cx, cy) = atom.center;
        let r = atom.radius;
        Some(match acc {
            None => (cx - r, cy - r, cx + r, cy + r),
            Some((min_x, min_y, max_x, max_y)) => (
                min_x.min(cx - r),
                min_y.min(cy - r),
                max_x.max(cx + r),
                max_y.max(cy + r),
            ),
        })
    })
}

/// Creates a circular polygon from a center point and radius
fn create_circle(center: &Point<f64>, radius: f64) -> Polygon<f64> {
    let num_points = 32;
    let coords: Vec<Coord<f64>> = (0..=num_points)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64) / (num_points as f64);
            Coord {
                x: center.x() + radius * angle.cos(),
                y: center.y() + radius * angle.sin(),
            }
        })
        .collect();

    Polygon::new(LineString(coords), vec![])
}

/// Creates an SVG path element from a polygon
fn create_svg_path(polygon: &Polygon<f64>, fill: &str, stroke: &str, stroke_width: f64) -> Path {
    Path::new()
        .set("fill", fill)
        .set("stroke", stroke)
        .set("stroke-width", stroke_width)
        .set("d", polygon_to_path_data(polygon))
}

fn polygon_to_path_data(polygon: &Polygon<f64>) -> String {
    let mut path_data = String::new();

    // Handle exterior ring
    let exterior = polygon.exterior();
    if let Some(first) = exterior.points().next() {
        path_data.push_str(&format!("M {} {} ", first.x(), first.y()));

        for point in exterior.points().skip(1) {
            path_data.push_str(&format!("L {} {} ", point.x(), point.y()));
        }
    }
    path_data.push('Z');

    // Handle interior rings (holes)
    for interior in polygon.interiors() {
        if let Some(first) = interior.points().next() {
            path_data.push_str(&format!("M {} {} ", first.x(), first.y()));

            for point in interior.points().skip(1) {
                path_data.push_str(&format!("L {} {} ", point.x(), point.y()));
            }
            path_data.push('Z');
        }
    }

    path_data
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferritin_core::load_model;
    use ferritin_test_data::TestFile;

    fn load_test_model() -> Model {
        let (path, _handle) = TestFile::protein_04().create_temp().unwrap();
        load_model(&path).unwrap()
    }

    #[test]
    fn test_flatten_structure_produces_svg() {
        let model = load_test_model();
        let doc = model.flatten_structure();
        let svg = doc.to_string();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"), "expected at least one residue path");
    }

    #[test]
    fn test_flatten_structure_respects_canvas_size() {
        let model = load_test_model();
        let options = FlattenOptions { canvas_size: 500.0, ..Default::default() };
        let doc = model.flatten_structure_with(&options);
        let svg = doc.to_string();
        assert!(svg.contains(r#"width="500""#));
        assert!(svg.contains(r#"height="500""#));
    }

    #[test]
    fn test_bounding_box_matches_atom_extent() {
        let atoms = vec![
            ProjectedAtom { chain_id: "A".into(), center: (0.0, 0.0), radius: 1.0 },
            ProjectedAtom { chain_id: "A".into(), center: (10.0, 5.0), radius: 2.0 },
        ];
        let (min_x, min_y, max_x, max_y) = bounding_box(atoms.iter()).unwrap();
        assert_eq!((min_x, min_y), (-1.0, -1.0));
        assert_eq!((max_x, max_y), (12.0, 7.0));
    }

    #[test]
    fn test_bounding_box_empty_is_none() {
        let atoms: Vec<ProjectedAtom> = vec![];
        assert!(bounding_box(atoms.iter()).is_none());
    }

    #[test]
    fn test_projection_axis_selects_correct_plane() {
        let coords = [1.0, 2.0, 3.0];
        assert_eq!(ProjectionAxis::Xy.project(coords), (1.0, 2.0));
        assert_eq!(ProjectionAxis::Xz.project(coords), (1.0, 3.0));
        assert_eq!(ProjectionAxis::Yz.project(coords), (2.0, 3.0));
    }

    #[test]
    fn test_flatten_structure_with_different_projections_differs() {
        let model = load_test_model();
        let xy = model
            .flatten_structure_with(&FlattenOptions { projection: ProjectionAxis::Xy, ..Default::default() })
            .to_string();
        let xz = model
            .flatten_structure_with(&FlattenOptions { projection: ProjectionAxis::Xz, ..Default::default() })
            .to_string();
        assert_ne!(xy, xz, "different projection axes should generally produce different outlines");
    }

    #[test]
    fn test_flatten_structure_uniform_fill_when_not_colored_by_chain() {
        let model = load_test_model();
        let options = FlattenOptions { color_by_chain: false, ..Default::default() };
        let doc = model.flatten_structure_with(&options);
        let svg = doc.to_string();
        assert!(svg.contains(r#"fill="blue""#));
        assert!(!svg.contains(CHAIN_PALETTE[0]));
    }
}
