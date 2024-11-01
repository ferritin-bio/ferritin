// use anyhow::{Context, Result};
// use geo::buffer::Buffer;
use geo::simplify::Simplify;
use geo::{LineString, Polygon};
use nalgebra::{Matrix2, Matrix3, Matrix4, Point2, Vector2, Vector3};
use ndarray::{s, Array1, Array2};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

// Helper Fns  ---------------------------------------------
//

pub fn orientation_from_topology(topologies: &[TopologyEntry]) -> bool {
    let mut first_ex = None;
    let mut first_cy = None;
    let mut first_he = None;

    for entry in topologies {
        match entry.description.as_str() {
            "Extracellular" if first_ex.is_none() => {
                first_ex = Some((entry.start, entry.end));
            }
            "Helical" if first_he.is_none() => {
                first_he = Some((entry.start, entry.end));
            }
            "Cytoplasmic" if first_cy.is_none() => {
                first_cy = Some((entry.start, entry.end));
            }
            _ => {}
        }
    }

    // Default to true (N->C orientation)
    match (first_ex, first_cy) {
        (Some((ex_start, _)), Some((cy_start, _))) => ex_start < cy_start,
        _ => true,
    }
}

pub fn orientation_from_ptm(ptm: &HashMap<String, (usize, usize)>) -> bool {
    match (ptm.get("chain"), ptm.get("signal peptide")) {
        (Some(&(chain_start, _)), Some(&(signal_start, _))) => signal_start < chain_start,
        _ => true,
    }
}

pub fn depth_slices_from_coord(xyz: &Array2<f64>, width: f64) -> Vec<Array2<f64>> {
    let z_coords = xyz.slice(s![.., 2]);
    let binned: Array1<i32> = (&z_coords / width).mapv(|x| x.floor() as i32);
    let min_bin = binned.fold(i32::MAX, |a, &b| a.min(b));
    let binned_shifted = &binned - min_bin;
    let num_bins = binned_shifted.fold(0, |a, &b| a.max(b)) + 1;

    let mut slice_coords = Vec::with_capacity(num_bins as usize);
    let mut total_coords = 0;

    for i in 0..num_bins {
        let mask = binned_shifted.mapv(|x| x == i);
        let bin_coords = xyz.slice(s![mask, ..]).to_owned();
        total_coords += bin_coords.nrows();
        slice_coords.push(bin_coords);
    }

    assert_eq!(xyz.nrows(), total_coords);
    slice_coords
}

/// Take an Nx3 coordinate matrix and return Z bin labels
pub fn get_z_slice_labels(xyz: &Array2<f64>, width: f64) -> Array1<i32> {
    let z_coords = xyz.slice(s![.., 2]);
    let binned: Array1<i32> = (&z_coords / width).mapv(|x| x.floor() as i32);
    let min_bin = binned.fold(i32::MAX, |a, &b| a.min(b));
    &binned - min_bin
}

/// Split matrix based on labels
pub fn split_on_labels(m: &Array2<f64>, labels: &Array1<i32>) -> Vec<Array2<f64>> {
    let num_bins = labels.fold(0, |a, &b| a.max(b)) + 1;
    let mut coords = Vec::with_capacity(num_bins as usize);
    let mut total_coords = 0;

    for i in 0..num_bins {
        let mask = labels.mapv(|x| x == i);
        let group_coords = m.slice(s![mask, ..]).to_owned();
        total_coords += group_coords.nrows();
        coords.push(group_coords);
    }

    assert_eq!(m.nrows(), total_coords);
    coords
}

/// Get dimensions from xy coordinates
pub fn get_dimensions(xy: &Array2<f64>, end_window: usize) -> HashMap<String, Value> {
    let mut dimensions = HashMap::new();

    // Width and height
    let x_coords = xy.column(0);
    let y_coords = xy.column(1);

    dimensions.insert(
        "width".to_string(),
        Value::Float(x_coords.max() - x_coords.min()),
    );
    dimensions.insert(
        "height".to_string(),
        Value::Float(y_coords.max() - y_coords.min()),
    );

    // Start and end means
    let start_mean = xy.slice(s![..end_window, ..]).mean();
    let end_mean = xy.slice(s![-end_window.., ..]).mean();
    dimensions.insert("start".to_string(), Value::Float(start_mean));
    dimensions.insert("end".to_string(), Value::Float(end_mean));

    // Bottom and top points
    let bottom_idx = y_coords
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    let top_idx = y_coords
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    dimensions.insert(
        "bottom".to_string(),
        Value::Point(Point2::new(xy[[bottom_idx, 0]], xy[[bottom_idx, 1]])),
    );
    dimensions.insert(
        "top".to_string(),
        Value::Point(Point2::new(xy[[top_idx, 0]], xy[[top_idx, 1]])),
    );

    dimensions
}

fn create_union_outline(coords: &Array2<f64>, radius: f64) -> Result<Polygon> {
    let points: Vec<_> = coords
        .rows()
        .into_iter()
        .map(|row| Point::new(row[0], row[1]))
        .collect();

    let buffers: Vec<_> = points.iter().map(|p| p.buffer(radius)).collect();

    Ok(union_polygons(&buffers)?)
}

fn create_back_outline(polygons: &[(HashMap<String, String>, Polygon)]) -> Result<Polygon> {
    let polys: Vec<_> = polygons.iter().map(|(_, p)| p.buffer(0.01)).collect();

    Ok(union_polygons(&polys)?)
}

fn composite_polygon(
    cartoon: &mut Cartoon,
    height_before: f64,
    height_after: f64,
    buffer_width: f64,
) {
    if height_before > 0.0 {
        let before_origin = cartoon.bottom_coord - Vector2::new(buffer_width, height_before);
        let before_poly = placeholder_polygon(height_before, buffer_width, before_origin);
        cartoon.styled_polygons.push(StyledPolygon {
            polygon: before_poly,
            facecolor: "#eeeeee".to_string(),
            shade: 0.5,
            edgecolor: "black".to_string(),
            linewidth: 1.0,
            zorder: -1,
        });
    }

    if height_after > 0.0 {
        let after_origin = cartoon.top_coord - Vector2::new(buffer_width, 0.0);
        let after_poly = placeholder_polygon(height_after, buffer_width, after_origin);
        cartoon.styled_polygons.push(StyledPolygon {
            polygon: after_poly,
            facecolor: "#eeeeee".to_string(),
            shade: 0.5,
            edgecolor: "black".to_string(),
            linewidth: 1.0,
            zorder: -1,
        });
    }

    cartoon.image_height += buffer_width + height_before + height_after;
    cartoon.bottom_coord -= Vector2::new(0.0, height_before);
    cartoon.top_coord += Vector2::new(0.0, height_after);
}

fn hls_to_rgb(h: f64, l: f64, s: f64) -> (f64, f64, f64) {
    // Implementation needed
    unimplemented!()
}

// Equivalent to get_sequential_colors
fn get_sequential_colors(colormap_name: &str, n: usize) -> Vec<RgbaColor> {
    // This would need a proper color map implementation
    // For now, returning a placeholder
    vec![
        RgbaColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0
        };
        n
    ]
}

fn export_placeholder(height: f64, name: &str, fname: &str, buffer_width: f64) {
    let poly = placeholder_polygon(height, buffer_width, Vector2::new(buffer_width, 0.0));
    let styled_polygons = vec![StyledPolygon {
        polygon: poly,
        facecolor: "#eeeeee".to_string(),
        shade: 0.5,
        edgecolor: "black".to_string(),
        linewidth: 1.0,
        zorder: -1,
    }];

    let data = PlaceholderData {
        polygons: styled_polygons,
        name: name.to_string(),
        width: buffer_width * 2.0,
        height: height + buffer_width,
        start: Vector2::new(buffer_width, 0.0),
        end: Vector2::new(height + 2.0 * buffer_width, 0.0),
        bottom: Vector2::new(buffer_width, 0.0),
        top: Vector2::new(height + 2.0 * buffer_width, 0.0),
    };

    let serialized = serde_json::to_string(&data).unwrap();
    let mut file = File::create(format!("{}.json", fname)).unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}

/// core outline creation.
///
///  args --> structure --> cartoon --> outline
pub fn make_cartoon(args: CartoonArgs) -> Result<()> {
    // Convert chain argument to single string if necessary
    let chain = if args.chain.len() == 1 {
        args.chain[0].clone()
    } else {
        args.chain.join("")
    };

    // Create molecule structure
    let mut molecule = Structure::new(
        &args.pdb,
        &chain,
        args.model,
        args.uniprot.as_deref(),
        false,
    )
    .context("Failed to create structure")?;

    // Handle view matrix
    if let Some(view_path) = args.view {
        let file = File::open(&view_path).context("Failed to open view file")?;
        let mut reader = BufReader::new(file);
        let mut first_line = String::new();
        reader
            .read_line(&mut first_line)
            .context("Failed to read first line of view file")?;

        if first_line.starts_with("set_view") {
            molecule
                .load_pymol_view(&view_path)
                .context("Failed to load PyMOL view")?;
        } else if first_line.starts_with("Model") {
            molecule
                .load_chimera_view(&view_path)
                .context("Failed to load Chimera view")?;
        } else {
            molecule
                .load_view_matrix(&view_path)
                .context("Failed to load view matrix")?;
        }
    } else {
        // Default to identity matrix if no view provided
        molecule.view_matrix = Matrix3::identity();
    }

    // Create cartoon outline
    let mut cartoon = molecule
        .outline(OutlineParams {
            outline_by: args.outline_by.clone(),
            depth: args.depth,
            radius: args.radius,
            only_annotated: args.only_annotated,
            only_ca: args.only_ca,
            depth_contour_interval: args.depth_contour_interval,
            back_outline: args.back_outline,
        })
        .context("Failed to create cartoon outline")?;

    // Determine color_residues_by
    let color_residues_by = if args.outline_by == "residue" && args.color_by != "same" {
        Some(args.color_by.clone())
    } else {
        None
    };

    // Determine colors
    let colors = if !args.colors.is_empty() {
        Some(args.colors.clone())
    } else {
        None
    };

    // Plot cartoon
    cartoon
        .plot(PlotOptions {
            do_show: false,
            axes_labels: args.axes,
            colors: colors.map(ColorScheme::from),
            color_residues_by,
            dpi: args.dpi,
            save: Some(args.save.clone()),
            depth_shading: args.depth_shading,
            depth_lines: args.depth_lines,
            edge_color: args.edge_color,
            line_width: args.line_width,
            ..Default::default()
        })
        .context("Failed to plot cartoon")?;

    // Export if requested
    if args.export {
        let export_path = Path::new(&args.save)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
            .to_string();

        cartoon
            .export(&export_path)
            .context("Failed to export cartoon")?;
    }

    Ok(())
}

pub fn matrix_from_nglview(m: &[f64]) -> (Matrix3<f64>, Vector3<f64>) {
    let camera_matrix = Matrix4::from_iterator(m.iter().cloned());
    let rotation = camera_matrix.fixed_slice::<3, 3>(0, 0);
    let translation = Vector3::new(
        camera_matrix[(3, 0)],
        camera_matrix[(3, 1)],
        camera_matrix[(3, 2)],
    );

    // Normalize rotation matrix
    let norms: Vec<f64> = (0..3).map(|i| rotation.row(i).norm()).collect();

    let normalized_rotation = Matrix3::from_fn(|i, j| rotation[(i, j)] / norms[i]);

    (normalized_rotation, translation)
}

pub fn matrix_to_nglview(m: &Matrix3<f64>) -> Vec<f64> {
    let mut nglv_matrix = Matrix4::identity();
    let transform = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);

    let rotated = m * transform;
    nglv_matrix
        .fixed_slice_mut::<3, 3>(0, 0)
        .copy_from(&rotated);

    nglv_matrix.as_slice().to_vec()
}

fn placeholder_polygon(height: f64, buffer_width: f64, origin: Vector2<f64>) -> Polygon<f64> {
    let line = LineString::from(vec![
        (buffer_width + origin.x, origin.y),
        (buffer_width + origin.x, height + origin.y),
    ]);
    line.buffer(buffer_width)
}

fn plot_polygon(
    poly: &Polygon<f64>,
    style: PolygonStyle,
    drawing_area: &DrawingArea<BitMapBackend, Cartesian2d<f64, f64>>,
    transform_params: TransformParams,
) -> Result<(), Box<dyn Error>> {
    match poly {
        poly if poly.area() > style.min_area => {
            let path = polygon_to_path(
                poly,
                style.min_interior_length,
                transform_params.translate_pre,
                transform_params.translate_post,
                transform_params.scale,
                transform_params.flip,
            );

            drawing_area.draw(
                &path
                    .styled()
                    .fill(style.facecolor)
                    .stroke_width(style.linewidth)
                    .stroke_color(style.edgecolor),
            )?;
        }
        _ => {}
    }
    Ok(())
}

fn polygon_to_path(
    polygon: &Polygon<f64>,
    min_interior_length: f64,
    translate_pre: Option<Vector2<f64>>,
    translate_post: Option<Vector2<f64>>,
    scale: f64,
    flip: bool,
) -> Path {
    let filtered_interiors: Vec<_> = polygon
        .interiors()
        .iter()
        .filter(|x| x.length() > min_interior_length)
        .collect();

    // Collect exterior vertices
    let mut vertices = polygon.exterior().points().collect::<Vec<_>>();

    // Add interior vertices
    for interior in filtered_interiors.iter() {
        vertices.extend(interior.points());
    }

    // Generate codes
    let mut codes = vec![];
    codes.extend(ring_coding(
        &polygon.exterior().points().collect::<Vec<_>>(),
    ));
    for interior in filtered_interiors.iter() {
        codes.extend(ring_coding(&interior.points().collect::<Vec<_>>()));
    }

    // Transform vertices
    let transformed_vertices: Vec<Point2<f64>> = vertices
        .iter()
        .map(|v| {
            let transformed = transform_coord(
                Vector2::new(v.x(), v.y()),
                translate_post,
                translate_pre,
                scale,
                flip,
            );
            Point2::new(transformed.x, transformed.y)
        })
        .collect();

    Path::new(transformed_vertices, codes)
}

// Helper functions for color conversion
fn rgb_to_hls(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // Implementation needed
    unimplemented!()
}

fn ring_coding(coordinates: &[(f64, f64)]) -> Vec<PathCode> {
    let mut codes = vec![PathCode::LineTo; coordinates.len()];
    if !coordinates.is_empty() {
        codes[0] = PathCode::MoveTo;
    }
    codes
}

// Equivalent to scale_line_width
fn scale_line_width(x: f64, lw_min: f64, lw_max: f64) -> f64 {
    lw_max * (1.0 - x) + lw_min * x
}

fn shade_from_color(color: RgbaColor, x: f64, range: f64) -> (f64, f64, f64) {
    let (h, l, s) = rgb_to_hls(color.r, color.g, color.b);
    let l_dark = (l - range / 2.0).max(0.0);
    let l_light = (l + range / 2.0).min(1.0);
    let l_new = l_dark * (1.0 - x) + l_light * x;
    hls_to_rgb(h, l_new, s)
}

// Equivalent to smooth_polygon
fn smooth_polygon(polygon: &Polygon<f64>, level: i32) -> Polygon<f64> {
    match level {
        0 => polygon
            .simplify(&0.3)
            .buffer(-2.0, 16, 1)
            .buffer(3.0, 16, 1),
        1 => polygon
            .simplify(&1.0)
            .buffer(3.0, 16, 1)
            .buffer(-5.0, 16, 1)
            .buffer(4.0, 16, 1),
        2 => polygon
            .simplify(&3.0)
            .buffer(5.0, 16, 1)
            .buffer(-9.0, 16, 1)
            .buffer(5.0, 16, 1),
        3 => polygon.simplify(&0.1).buffer(2.0, 16, 1),
        _ => polygon.clone(),
    }
}

fn transform_coord(
    xy: Vector2<f64>,
    translate_post: Option<Vector2<f64>>,
    translate_pre: Option<Vector2<f64>>,
    scale: f64,
    flip: bool,
) -> Vector2<f64> {
    let mut xy = xy;

    if let Some(pre_trans) = translate_pre {
        xy += pre_trans;
    }

    if flip {
        let flip_matrix = Matrix2::new(-1.0, 0.0, 0.0, -1.0);
        xy = flip_matrix * xy;
    }

    if let Some(post_trans) = translate_post {
        xy += post_trans;
    }

    xy * scale
}

// ENUMS  ---------------------------------------------

#[derive(Clone)]
enum ColorScheme {
    Single(RGBColor),
    Multiple(Vec<RGBColor>),
    Named(String),
    Map(HashMap<String, RGBColor>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PathCode {
    MoveTo,
    LineTo,
}

// Structs  ---------------------------------------------

#[derive(Debug, Clone)]
pub struct Cartoon {
    name: String,
    polygons: Vec<(ResidueInfo, Polygon<f64>)>,
    residues: Vec<ResidueInfo>,
    residues_flat: Vec<ResidueInfo>,
    outline_by: String,
    num_groups: usize,
    groups: Vec<String>,
    back_outline: Option<Polygon<f64>>,
    group_outlines: Vec<Polygon<f64>>,
    dimensions: Vector2<f64>,
    styled_polygons: Vec<StyledPolygon>,
    image_width: f64,
    image_height: f64,
    bottom_coord: Vector2<f64>,
    top_coord: Vector2<f64>,
}

impl Cartoon {
    pub fn new(
        name: String,
        polygons: Vec<(ResidueInfo, Polygon<f64>)>,
        residues: Vec<ResidueInfo>,
        outline_by: String,
        back_outline: Option<Polygon<f64>>,
        group_outlines: Vec<Polygon<f64>>,
        num_groups: usize,
        dimensions: Vector2<f64>,
        groups: Vec<String>,
    ) -> Self {
        Cartoon {
            name,
            polygons,
            residues_flat: residues,
            outline_by,
            num_groups,
            groups,
            back_outline,
            group_outlines,
            dimensions,
            styled_polygons: Vec::new(),
            image_width: dimensions.x,
            image_height: dimensions.y,
        }
    }

    pub fn plot(&mut self, plot_options: PlotOptions) -> Result<(), Box<dyn Error>> {
        // Implementation of plot function
        // This would be a long implementation similar to the Python version
        // but using the plotters crate for rendering

        // Create styled polygons list
        self.styled_polygons = Vec::new();

        // Create drawing area
        let root = BitMapBackend::new(&plot_options.output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Setup coordinate system
        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(0f64..self.dimensions.x, 0f64..self.dimensions.y)?;

        // Plot background outline if exists
        if let Some(back_outline) = &self.back_outline {
            // Plot background outline
        }

        // Plot group outlines
        for outline in &self.group_outlines {
            // Plot group outlines
        }

        // Plot main polygons
        for (info, polygon) in &self.polygons {
            // Plot individual polygons with styling
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CartoonArgs {
    pub pdb: String,
    pub chain: Vec<String>,
    pub model: usize,
    pub uniprot: Option<String>,
    pub view: Option<String>,
    pub outline_by: String,
    pub depth: bool,
    pub radius: f64,
    pub only_annotated: bool,
    pub only_ca: bool,
    pub depth_contour_interval: f64,
    pub back_outline: bool,
    pub color_by: String,
    pub colors: Vec<String>,
    pub axes: bool,
    pub dpi: u32,
    pub save: String,
    pub depth_shading: bool,
    pub depth_lines: bool,
    pub edge_color: String,
    pub line_width: f64,
    pub export: bool,
}

#[derive(Serialize, Deserialize)]
struct StyledPolygon {
    polygon: Polygon<f64>,
    facecolor: String,
    shade: f64,
    edgecolor: String,
    linewidth: f64,
    zorder: i32,
}

#[derive(Serialize, Deserialize)]
struct PlaceholderData {
    polygons: Vec<StyledPolygon>,
    name: String,
    width: f64,
    height: f64,
    start: Vector2<f64>,
    end: Vector2<f64>,
    bottom: Vector2<f64>,
    top: Vector2<f64>,
}

// Supporting structures
#[derive(Clone)]
struct PlotOptions {
    colors: Option<ColorScheme>,
    axes_labels: bool,
    color_residues_by: Option<String>,
    edge_color: RGBColor,
    line_width: f64,
    depth_shading: bool,
    depth_lines: bool,
    shading_range: f64,
    smoothing: bool,
    output_path: String,
    dpi: u32,
    placeholder: Option<f64>,
}

// Equivalent to shade_from_color
// Note: This requires a color handling library
#[derive(Debug, Clone, Copy)]
struct RgbaColor {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}

#[derive(Clone)]
struct StyledPolygon {
    polygon: Polygon<f64>,
    facecolor: RGBColor,
    edgecolor: RGBColor,
    linewidth: f64,
    shade: Option<f64>,
    base_fc: RGBColor,
}

#[derive(Debug)]
pub struct OutlineOptions {
    pub by: String,
    pub depth: Option<String>,
    pub depth_contour_interval: f64,
    pub only_backbone: bool,
    pub only_ca: bool,
    pub only_annotated: bool,
    pub radius: Option<f64>,
    pub back_outline: bool,
    pub align_transmembrane: bool,
}

impl Default for PlotOptions {
    fn default() -> Self {
        PlotOptions {
            do_show: true,
            axes_labels: false,
            colors: None,
            color_residues_by: None,
            dpi: 300,
            save: None,
            depth_shading: false,
            depth_lines: false,
            edge_color: String::from("black"),
            line_width: 0.7,
            smoothing: false,
            placeholder: None,
        }
    }
}

impl ColorScheme {
    fn from(colors: Vec<String>) -> Self {
        ColorScheme::Multiple(colors)
    }
}

#[derive(Debug, Clone)]
pub struct TopologyEntry {
    pub description: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct ResidueInfo {
    chain: String,
    id: i32,
    amino_acid: String,
    object: Residue,
    coord: (usize, usize),
    coord_ca: (usize, usize),
    coord_backbone: Vec<usize>,
    atoms: Vec<String>,
    domain: Option<String>,
    topology: Option<String>,
    xyz: Option<Array2<f64>>,
    polygon: Option<Polygon>,
    depth: Option<f64>,
}
impl ResidueInfo {
    fn to_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        if let Some(depth) = self.depth {
            metadata.insert("depth".to_string(), depth.to_string());
        }
        if let Some(domain) = &self.domain {
            metadata.insert("domain".to_string(), domain.clone());
        }
        if let Some(topology) = &self.topology {
            metadata.insert("topology".to_string(), topology.clone());
        }
        metadata.insert("chain".to_string(), self.chain.clone());
        metadata
    }

    fn get_group_key(&self, by: &str) -> Result<String> {
        match by {
            "chain" => Ok(self.chain.clone()),
            "domain" => Ok(self.domain.clone().unwrap_or_else(|| "None".to_string())),
            "topology" => Ok(self.topology.clone().unwrap_or_else(|| "None".to_string())),
            _ => Err(anyhow::anyhow!("Invalid grouping key")),
        }
    }
}

/// Structure class for loading coordinates and generating cartoons
pub struct Structure {
    name: String,
    structure: BioStructure,
    chains: Vec<String>,
    is_opm: bool,
    use_nglview: bool,
    view_matrix: Option<Matrix3<f64>>,
    residues: HashMap<String, HashMap<i32, ResidueInfo>>,
    residues_flat: Vec<ResidueInfo>,
    sequence: HashMap<String, String>,
    coord: Array2<f64>,
    rotated_coord: Option<Array2<f64>>,
    ca_atoms: Vec<usize>,
    backbone_atoms: Vec<usize>,
    uniprot: Option<UniProtRecord>,
    uniprot_xml: Option<String>,
    uniprot_overlap: Option<Array1<i32>>,
    uniprot_offset: Option<i32>,
}

impl Structure {
    pub fn new(
        file: &str,
        name: Option<&str>,
        model: usize,
        chain: &str,
        uniprot: Option<&str>,
        view: bool,
        is_opm: bool,
        res_start: Option<i32>,
        res_end: Option<i32>,
    ) -> Result<Self> {
        // Parse file extension and choose appropriate parser
        let file_path = Path::new(file);
        let (mut pdb, error) = pdbtbx::open(file_path);

        // let parser = match ext.as_str() {
        //     "cif" | "mcif" => bio::structure::MMCIFParser::new(),
        //     "pdb" | "ent" => bio::structure::PDBParser::new(),
        //     _ => return Err(anyhow::anyhow!("File format not recognized!")),
        // };

        // let structure = parser.parse_file(file)?;
        // let model = structure.models().nth(model).context("Model not found")?;

        // Handle chain selection
        let all_chains: Vec<_> = model.chains().map(|c| c.id().to_string()).collect();

        let chains = if chain.to_lowercase() == "all" {
            all_chains.clone()
        } else {
            chain.chars().map(|c| c.to_string()).collect()
        };

        // Initialize data structures
        let mut residues = HashMap::new();
        let mut sequence = HashMap::new();
        let mut coord = Vec::new();
        let mut ca_atoms = Vec::new();
        let mut backbone_atoms = Vec::new();
        let mut all_atoms = 0;

        // Process each chain
        for chain_id in &chains {
            sequence.insert(chain_id.clone(), String::new());
            residues.insert(chain_id.clone(), HashMap::new());

            if let Some(chain) = model.chain(chain_id) {
                for residue in chain.residues() {
                    let res_id = residue.number();

                    // Skip HETATM records and non-standard amino acids
                    if residue.het_flag() || !AMINO_ACID_3LETTER.contains_key(residue.name()) {
                        continue;
                    }

                    let res_aa = AMINO_ACID_3LETTER[residue.name()].to_string();
                    sequence.get_mut(chain_id).unwrap().push_str(&res_aa);

                    let mut residue_atoms = 0;
                    let mut these_atoms = Vec::new();
                    let mut backbone_atoms_local = Vec::new();
                    let mut this_ca_atom = 0;

                    for atom in residue.atoms() {
                        coord.push(atom.position().coords);
                        these_atoms.push(atom.name().to_string());

                        if atom.name() == "CA" {
                            this_ca_atom = all_atoms;
                            ca_atoms.push(all_atoms);
                        }

                        if ["CA", "N", "C", "O"].contains(&atom.name()) {
                            backbone_atoms_local.push(all_atoms);
                            backbone_atoms.push(all_atoms);
                        }

                        all_atoms += 1;
                        residue_atoms += 1;
                    }

                    // Create ResidueInfo
                    residues.get_mut(chain_id).unwrap().insert(
                        res_id,
                        ResidueInfo {
                            chain: chain_id.clone(),
                            id: res_id,
                            amino_acid: res_aa,
                            object: residue.clone(),
                            coord: (all_atoms - residue_atoms, all_atoms),
                            coord_ca: (this_ca_atom, this_ca_atom + 1),
                            coord_backbone: backbone_atoms_local,
                            atoms: these_atoms,
                            domain: None,
                            topology: None,
                            xyz: None,
                            polygon: None,
                            depth: None,
                        },
                    );
                }
            }
        }

        let coord = Array2::from_shape_vec((all_atoms, 3), coord.into_iter().flatten().collect())?;

        let mut structure = Structure {
            name: name
                .unwrap_or_else(|| file_path.file_name().unwrap().to_str().unwrap())
                .to_string(),
            structure,
            chains,
            is_opm,
            use_nglview: view,
            view_matrix: None,
            residues,
            residues_flat: Vec::new(),
            sequence,
            coord,
            rotated_coord: None,
            ca_atoms,
            backbone_atoms,
            uniprot: None,
            uniprot_xml: None,
            uniprot_overlap: None,
            uniprot_offset: None,
        };

        // Handle UniProt data if provided
        if let Some(uniprot_id) = uniprot {
            structure.process_uniprot(uniprot_id)?;
        }

        Ok(structure)
    }
    pub fn outline(&mut self, options: OutlineOptions) -> Result<Cartoon> {
        // Validate options
        if !["all", "residue", "chain", "domain", "topology"].contains(&options.by.as_str()) {
            return Err(anyhow::anyhow!("Invalid 'by' option"));
        }
        if let Some(depth) = &options.depth {
            if !["flat", "contours"].contains(&depth.as_str()) {
                return Err(anyhow::anyhow!("Invalid depth option"));
            }
        }

        // Flatten residue hierarchy
        self.residues_flat = self
            .residues
            .iter()
            .flat_map(|(_, chain_residues)| chain_residues.values().cloned())
            .collect();

        // // Update view matrix based on configuration
        // if self.is_opm {
        //     self.set_view_matrix(Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0))?;
        // } else if self.use_nglview {
        //     self.update_view_matrix()?;
        // }

        // // Transform coordinates
        // self.apply_view_matrix()?;

        // // Recenter coordinates
        // let offset_x = self.rotated_coord.as_ref().unwrap().column(0).min()?;
        // let offset_y = if self.is_opm {
        //     0.0
        // } else {
        //     self.rotated_coord.as_ref().unwrap().column(1).min()?
        // };

        if let Some(mut coords) = self.rotated_coord.take() {
            coords -= Array2::from_shape_vec((1, 3), vec![offset_x, offset_y, 0.0])?;
            self.rotated_coord = Some(coords);
        }

        // Handle transmembrane alignment if requested
        if options.align_transmembrane && self.uniprot_xml.is_some() {
            self.align_transmembrane()?;
        }

        // Initialize containers for results
        let mut polygons = Vec::new();
        let mut groups = HashMap::new();
        self.group_outlines = Vec::new();

        // Determine radius for atom representation
        let radius = match (options.radius, options.only_ca, options.only_backbone) {
            (Some(r), _, _) => r,
            (None, true, _) => 5.0,
            (None, _, true) => 4.0,
            (None, _, _) => 1.5,
        };

        // Process based on grouping method
        match options.by.as_str() {
            "all" => self.outline_all(&mut polygons, radius, &options)?,
            "residue" => self.outline_by_residue(&mut polygons, radius, &options)?,
            _ => self.outline_by_group(&mut polygons, &mut groups, radius, &options)?,
        }

        // Create back outline if requested
        if options.back_outline {
            self.back_outline = Some(create_back_outline(&polygons)?);
        } else {
            self.back_outline = None;
        }

        // Create and return Cartoon
        Ok(Cartoon::new(
            self.name.clone(),
            polygons,
            self.residues_flat.clone(),
            options.by,
            self.back_outline.clone(),
            self.group_outlines.clone(),
            self.num_groups,
            get_dimensions(&self.rotated_coord.as_ref().unwrap())?,
            groups.keys().cloned().collect(),
        ))
    }

    fn outline_all(
        &mut self,
        polygons: &mut Vec<(HashMap<String, String>, Polygon)>,
        radius: f64,
        options: &OutlineOptions,
    ) -> Result<()> {
        self.num_groups = 1;
        let coords = if options.only_ca {
            self.get_ca_coordinates()?
        } else if options.only_backbone {
            self.get_backbone_coordinates()?
        } else {
            self.rotated_coord.as_ref().unwrap().clone()
        };

        if let Some(depth) = &options.depth {
            if depth == "contours" {
                let slice_labels = get_z_slice_labels(&coords, options.depth_contour_interval);
                let slices = split_on_labels(&coords, &slice_labels);

                for slice in slices {
                    let slice_depth = self.calculate_z_scale(slice.column(2).mean()?);
                    let outline = create_union_outline(&slice, radius)?;
                    let mut metadata = HashMap::new();
                    metadata.insert("depth".to_string(), slice_depth.to_string());
                    polygons.push((metadata, outline));
                }
            }
        } else {
            let outline = create_union_outline(&coords, radius)?;
            polygons.push((HashMap::new(), outline));
        }
        Ok(())
    }

    fn outline_by_residue(
        &mut self,
        polygons: &mut Vec<(HashMap<String, String>, Polygon)>,
        radius: f64,
        options: &OutlineOptions,
    ) -> Result<()> {
        self.num_groups = 1;

        // Sort residues by Z-depth
        let mut residues = self.residues_flat.clone();
        residues.sort_by(|a, b| {
            let a_depth = a.xyz.as_ref().unwrap().column(2).mean().unwrap_or(0.0);
            let b_depth = b.xyz.as_ref().unwrap().column(2).mean().unwrap_or(0.0);
            a_depth.partial_cmp(&b_depth).unwrap()
        });

        for res in residues {
            let outline = create_union_outline(&res.xyz.unwrap(), radius)?;
            let depth = self.calculate_z_scale(res.xyz.unwrap().column(2).mean()?);
            let mut res_info = res.clone();
            res_info.polygon = Some(outline.clone());
            res_info.depth = Some(depth);
            polygons.push((res_info.to_metadata(), outline));
        }
        Ok(())
    }

    fn outline_by_group(
        &mut self,
        polygons: &mut Vec<(HashMap<String, String>, Polygon)>,
        groups: &mut HashMap<String, Vec<ResidueInfo>>,
        radius: f64,
        options: &OutlineOptions,
    ) -> Result<()> {
        if options.by == "domain" || options.by == "topology" {
            if self.uniprot_xml.is_none() {
                return Err(anyhow::anyhow!(
                    "UniProt data required for domain/topology grouping"
                ));
            }
        }

        // Group residues
        for res in &self.residues_flat {
            let group_key = res.get_group_key(&options.by)?;
            groups
                .entry(group_key)
                .or_insert_with(Vec::new)
                .push(res.clone());
            f
        }

        self.num_groups = groups.len();

        if let Some(depth) = &options.depth {
            self.process_depth_grouped_outlines(polygons, groups, radius, options, depth)?;
        } else {
            self.process_simple_grouped_outlines(polygons, groups, radius, options)?;
        }

        Ok(())
    }

    // // Private helper methods
    // fn process_structure(&mut self, res_start: Option<i32>, res_end: Option<i32>) -> Result<()> {
    //     // Mock implementation
    //     Ok(())
    // }

    // fn process_uniprot(&mut self, uniprot_id: &str) -> Result<()> {
    //     // Mock implementation
    //     Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_line_width() {
        assert_relative_eq!(scale_line_width(0.0, 1.0, 2.0), 2.0);
        assert_relative_eq!(scale_line_width(1.0, 1.0, 2.0), 1.0);
        assert_relative_eq!(scale_line_width(0.5, 1.0, 2.0), 1.5);
    }
}
