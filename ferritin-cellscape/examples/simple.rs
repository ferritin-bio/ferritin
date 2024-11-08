use geo::{BooleanOps, Coord, LineString, MultiPolygon, Point, Polygon};
use std::error::Error;
use std::f64::consts::PI;
use svg::node::element::{Circle, Path};
use svg::Document;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Generate some test points (x, y, radius)
    let points = vec![
        (100.0, 100.0, 30.0),
        (150.0, 120.0, 25.0),
        (120.0, 150.0, 20.0),
        (100.0, 130.0, 20.0),
    ];

    // 2. Create circles
    let circles: Vec<Polygon<f64>> = points
        .iter()
        .map(|(x, y, r)| create_circle(&Point::new(*x, *y), *r))
        .collect();

    // 3. Merge circles together
    let merged = circles
        .iter()
        .fold(MultiPolygon(vec![circles[0].clone()]), |acc, circle| {
            acc.union(&circle.clone())
        });

    // 4. Create SVG
    let margin = 10.0;
    let mut document = Document::new().set("width", 300).set("height", 300).set(
        "viewBox",
        (-margin, -margin, 300.0 + margin, 300.0 + margin),
    );

    // Add paths for each polygon in the multipolygon
    for polygon in merged.0 {
        document = document.add(create_svg_path(&polygon, "blue", "black", 2.0));
    }

    // 5. Save SVG
    svg::save("circles.svg", &document)?;
    println!("SVG has been created as 'circles.svg'");

    Ok(())
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

/// Converts a polygon to SVG path data
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

/// Creates an SVG path element from a polygon
fn create_svg_path(polygon: &Polygon<f64>, fill: &str, stroke: &str, stroke_width: f64) -> Path {
    Path::new()
        .set("fill", fill)
        .set("stroke", stroke)
        .set("stroke-width", stroke_width)
        .set("d", polygon_to_path_data(polygon))
}

/// Creates SVG circles for the original points
fn create_point_circles(points: &[(f64, f64, f64)]) -> svg::node::element::Group {
    let mut group = svg::node::element::Group::new().set("class", "points");

    for (x, y, r) in points {
        let circle = Circle::new()
            .set("cx", *x)
            .set("cy", *y)
            .set("r", *r)
            .set("fill", "rgba(0, 0, 255, 0.1)")
            .set("stroke", "blue")
            .set("stroke-width", 0.5);
        group = group.add(circle);
    }

    group
}
