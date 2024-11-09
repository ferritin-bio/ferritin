//! Core Cellscape Namespace
//!
//!
//! ** WORK IN PROGRESS **
//!
use ferritin_core::AtomCollection;
use geo::{BooleanOps, Coord, LineString, MultiPolygon, Point, Polygon};
use std::f64::consts::PI;
use svg::node::element::Path;
use svg::Document;

// Traits -------------------------------------------------------------------------------------

/// Flattens 3D atomic structures into a 2D SVG representation.
pub trait StructureFlatten {
    fn flatten_structure(&self) -> Document;
}

// Helper Functions ---------------------------------------------------------------------------

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
// IMPL---------------------------------------------------------------------------

impl StructureFlatten for AtomCollection {
    fn flatten_structure(&self) -> Document {
        let margin = 10.0;
        let radius = 50.;

        let merged_polygons: Vec<Polygon<f64>> = self
            .iter_residues_aminoacid()
            .flat_map(|residue| {
                // Get atom coordinates and create circles
                let circles: Vec<Polygon<f64>> = residue
                    .iter_atoms()
                    .map(|atm| {
                        let [x, y, _] = atm.coords;
                        create_circle(&Point::new(*x as f64, *y as f64), radius)
                    })
                    .collect();

                // Skip empty residues
                if circles.is_empty() {
                    return vec![];
                }

                // Merge circles into outline
                circles
                    .iter()
                    .fold(MultiPolygon(vec![circles[0].clone()]), |acc, circle| {
                        acc.union(&circle.clone())
                    })
                    .0
            })
            .collect();

        let base_document = Document::new().set("width", 300).set("height", 300).set(
            "viewBox",
            (-margin, -margin, 300.0 + margin, 300.0 + margin),
        );

        merged_polygons.iter().fold(base_document, |doc, polygon| {
            doc.add(create_svg_path(polygon, "blue", "black", 2.0))
        })
    }
}
