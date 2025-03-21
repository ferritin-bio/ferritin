//! Structure.
//!
//! Struct for rendering with Bevy
//!
//!

use super::ColorScheme;
use bevy::log::tracing_subscriber::reload::Error;
use bevy::math::Vec4;
use bevy::prelude::{
    Color, Component, Cylinder, Mesh, MeshBuilder, Meshable, Quat, Sphere, StandardMaterial,
    Transform, Vec3, default,
};
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bon::Builder;
use ferritin_core::AtomCollection;

/// Enum representing various rendering options.
///
/// Each of these enums represents a rendering path that can be used by a `Structure`
///
/// Down the Line: allow passing an arbitrary function that maps PDB to mesh.
///
#[derive(Clone)]
pub enum RenderOptions {
    Wireframe,
    Cartoon,
    BallAndStick,
    Solid,
    Putty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SecondaryStructure {
    Helix,
    Sheet,
    Loop,
}

// Structure to hold residue information needed for cartoon rendering
struct BackboneAtoms {
    ca: Vec3, // Alpha carbon
    n: Vec3,  // Nitrogen
    c: Vec3,  // Carbonyl carbon
    o: Vec3,  // Carbonyl oxygen (for orientation)
    residue_index: usize,
}

/// Define Everything Needed to render
#[derive(Builder, Component)]
pub struct Structure {
    pdb: AtomCollection,
    #[builder(default = RenderOptions::Solid)]
    rendertype: RenderOptions,
    #[builder(default = ColorScheme::Solid(Color::WHITE))]
    color_scheme: ColorScheme,
    #[builder(default = StandardMaterial::default())]
    material: StandardMaterial,
}

impl Structure {
    pub fn to_mesh(&self) -> Mesh {
        match &self.rendertype {
            RenderOptions::Wireframe => self.render_wireframe(),
            RenderOptions::Cartoon => self.render_cartoon(),
            RenderOptions::BallAndStick => self.render_ballandstick(),
            RenderOptions::Solid => self.render_spheres(),
            RenderOptions::Putty => self.render_putty().unwrap(),
        }
    }
    pub fn get_material(&self) -> StandardMaterial {
        self.material.clone()
    }
    fn create_smooth_curve(points: &[Vec3], segments: usize) -> Vec<Vec3> {
        let mut curve_points = Vec::new();

        for i in 0..points.len() - 1 {
            let p0 = if i == 0 { points[0] } else { points[i - 1] };
            let p1 = points[i];
            let p2 = points[i + 1];
            let p3 = if i + 2 >= points.len() {
                points[points.len() - 1]
            } else {
                points[i + 2]
            };

            for t in 0..segments {
                let t = t as f32 / segments as f32;
                let pos = Structure::catmull_rom(p0, p1, p2, p3, t);
                curve_points.push(pos);
            }
        }

        curve_points
    }

    /// Catmull-Rom spline interpolation
    fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
        let t2 = t * t;
        let t3 = t2 * t;
        let v0 = (p2 - p0) * 0.5;
        let v1 = (p3 - p1) * 0.5;

        (2.0 * p1 - 2.0 * p2 + v0 + v1) * t3
            + (-3.0 * p1 + 3.0 * p2 - 2.0 * v0 - v1) * t2
            + v0 * t
            + p1
    }

    // Function to extract backbone atoms for proper orientation
    fn extract_backbone_atoms(pdb: &AtomCollection) -> Vec<BackboneAtoms> {
        let mut backbone_data = Vec::new();

        for residue in pdb.iter_residues_aminoacid() {
            if let (Some(ca), Some(n), Some(c), Some(o)) = (
                residue.find_atom_by_name("CA"),
                residue.find_atom_by_name("N"),
                residue.find_atom_by_name("C"),
                residue.find_atom_by_name("O"),
            ) {
                backbone_data.push(BackboneAtoms {
                    ca: Vec3::from_array(*ca.coords()),
                    n: Vec3::from_array(*n.coords()),
                    c: Vec3::from_array(*c.coords()),
                    o: Vec3::from_array(*o.coords()),
                    residue_index: residue.residue_id() as usize,
                });
            }
        }

        backbone_data
    }

    // Function to generate alpha helix mesh
    fn generate_alpha_helix_mesh(backbone_atoms: &[BackboneAtoms], segment: &[usize]) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        let mut colors = Vec::new();

        // Parameters for alpha helix
        let radius = 2.3; // Helix radius (Å)
        let rise_per_residue = 1.5; // Rise per residue (Å)
        let residues_per_turn = 3.6; // Standard alpha helix
        let ribbon_width = 0.6; // Width of the ribbon (Å)
        let segments_per_residue = 4; // Smoothness

        // Generate the helix backbone
        let helix_atoms: Vec<&BackboneAtoms> = segment
            .iter()
            .filter_map(|idx| backbone_atoms.iter().find(|a| a.residue_index == *idx))
            .collect();

        if helix_atoms.len() < 2 {
            return Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        }

        // Create a smooth helical path
        let total_segments = helix_atoms.len() * segments_per_residue;
        for i in 0..total_segments {
            let t = i as f32 / total_segments as f32;
            let residue_idx = t * (helix_atoms.len() - 1) as f32;
            let angle = residue_idx * 2.0 * std::f32::consts::PI / residues_per_turn;

            // Interpolate position along the helix axis
            let base_pos = Structure::interpolate_position(&helix_atoms, residue_idx);

            // Calculate helix position with proper spiral
            let helix_x = radius * angle.cos();
            let helix_y = radius * angle.sin();
            let helix_z = rise_per_residue * residue_idx;

            // Transform to align with protein backbone
            let (axis, up) = Structure::calculate_helix_orientation(&helix_atoms, residue_idx);
            let right = axis.cross(up).normalize();

            // Generate ribbon vertices
            let pos_inner =
                base_pos + (right * helix_x + up * helix_y) * (radius - ribbon_width / 2.0);
            let pos_outer =
                base_pos + (right * helix_x + up * helix_y) * (radius + ribbon_width / 2.0);

            positions.push([pos_inner.x, pos_inner.y, pos_inner.z]);
            positions.push([pos_outer.x, pos_outer.y, pos_outer.z]);

            // Calculate normals (pointing outward from helix axis)
            let normal = (right * helix_x + up * helix_y).normalize();
            normals.push([normal.x, normal.y, normal.z]);
            normals.push([normal.x, normal.y, normal.z]);

            // Add colors - could be based on secondary structure
            let color = [1.0, 0.7, 0.7, 1.0]; // Example color for helix
            colors.push(color);
            colors.push(color);
        }

        // Generate indices for triangles
        for i in 0..total_segments - 1 {
            let i0 = i * 2;
            let i1 = i0 + 1;
            let i2 = i0 + 2;
            let i3 = i0 + 3;

            indices.push(i0 as u32);
            indices.push(i2 as u32);
            indices.push(i1 as u32);

            indices.push(i1 as u32);
            indices.push(i2 as u32);
            indices.push(i3 as u32);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            colors
                .iter()
                .map(|&c| Vec4::new(c[0], c[1], c[2], c[3]))
                .collect::<Vec<_>>(),
        );
        mesh.insert_indices(Indices::U32(indices));

        mesh
    }

    // Function to generate beta sheet mesh (flat arrow-shaped ribbon)
    fn generate_beta_sheet_mesh(backbone_atoms: &[BackboneAtoms], segment: &[usize]) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        let mut colors = Vec::new();

        // Parameters for beta sheet
        let ribbon_width = 1.5; // Width of beta strand (Å)
        let arrow_width = 2.5; // Width at arrow tip (Å)
        let segments_per_residue = 2; // Smoothness

        // Extract atoms for this sheet segment
        let sheet_atoms: Vec<&BackboneAtoms> = segment
            .iter()
            .filter_map(|idx| backbone_atoms.iter().find(|a| a.residue_index == *idx))
            .collect();

        if sheet_atoms.len() < 2 {
            return Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        }

        // Generate the sheet backbone and ribbon
        let total_segments = sheet_atoms.len() * segments_per_residue;
        for i in 0..total_segments {
            let t = i as f32 / (total_segments - 1) as f32;
            let residue_idx = t * (sheet_atoms.len() - 1) as f32;

            // Calculate ribbon width - wider at the end for arrow shape
            let is_arrow_tip = i >= total_segments - segments_per_residue;
            let width = if is_arrow_tip {
                arrow_width * (i - (total_segments - segments_per_residue)) as f32
                    / segments_per_residue as f32
                    + ribbon_width
                        * (1.0
                            - (i - (total_segments - segments_per_residue)) as f32
                                / segments_per_residue as f32)
            } else {
                ribbon_width
            };

            // Get backbone position
            let pos = Structure::interpolate_position(&sheet_atoms, residue_idx);

            // Calculate peptide plane orientation
            let (strand_dir, normal) =
                Structure::calculate_sheet_orientation(&sheet_atoms, residue_idx);
            let side_dir = strand_dir.cross(normal).normalize();

            // Generate ribbon vertices on both sides of backbone
            let half_width = width / 2.0;
            let pos_left = pos - side_dir * half_width;
            let pos_right = pos + side_dir * half_width;

            positions.push([pos_left.x, pos_left.y, pos_left.z]);
            positions.push([pos_right.x, pos_right.y, pos_right.z]);

            // Use peptide plane normal for both vertices
            normals.push([normal.x, normal.y, normal.z]);
            normals.push([normal.x, normal.y, normal.z]);

            // Add colors - could be based on secondary structure
            let color = [0.7, 0.7, 1.0, 1.0]; // Example color for sheet
            colors.push(color);
            colors.push(color);
        }

        // Generate indices for triangles
        for i in 0..total_segments - 1 {
            let i0 = i * 2;
            let i1 = i0 + 1;
            let i2 = i0 + 2;
            let i3 = i0 + 3;

            indices.push(i0 as u32);
            indices.push(i2 as u32);
            indices.push(i1 as u32);

            indices.push(i1 as u32);
            indices.push(i2 as u32);
            indices.push(i3 as u32);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            colors
                .iter()
                .map(|&c| Vec4::new(c[0], c[1], c[2], c[3]))
                .collect::<Vec<_>>(),
        );
        mesh.insert_indices(Indices::U32(indices));

        mesh
    }

    // Utility functions for orientation and interpolation
    fn interpolate_position(atoms: &[&BackboneAtoms], residue_idx: f32) -> Vec3 {
        let idx = residue_idx.floor() as usize;
        let frac = residue_idx - idx as f32;

        if idx >= atoms.len() - 1 {
            return atoms.last().unwrap().ca;
        }

        atoms[idx].ca * (1.0 - frac) + atoms[idx + 1].ca * frac
    }

    fn calculate_helix_orientation(atoms: &[&BackboneAtoms], residue_idx: f32) -> (Vec3, Vec3) {
        let idx = residue_idx.floor() as usize;

        if idx >= atoms.len() - 1 {
            return Structure::calculate_peptide_plane(atoms.last().unwrap());
        }

        let (axis1, up1) = Structure::calculate_peptide_plane(atoms[idx]);
        let (axis2, up2) = Structure::calculate_peptide_plane(atoms[idx + 1]);

        let frac = residue_idx - idx as f32;
        let axis = axis1 * (1.0 - frac) + axis2 * frac;
        let up = up1 * (1.0 - frac) + up2 * frac;

        (axis.normalize(), up.normalize())
    }

    fn calculate_sheet_orientation(atoms: &[&BackboneAtoms], residue_idx: f32) -> (Vec3, Vec3) {
        let idx = residue_idx.floor() as usize;

        if idx >= atoms.len() - 1 {
            return Structure::calculate_peptide_plane(atoms.last().unwrap());
        }

        // For beta sheets, we want the ribbon to be perpendicular to the peptide plane
        let (strand_dir1, normal1) = Structure::calculate_peptide_plane(atoms[idx]);
        let (strand_dir2, normal2) = Structure::calculate_peptide_plane(atoms[idx + 1]);

        let frac = residue_idx - idx as f32;
        let strand_dir = strand_dir1 * (1.0 - frac) + strand_dir2 * frac;
        let normal = normal1 * (1.0 - frac) + normal2 * frac;

        (strand_dir.normalize(), normal.normalize())
    }

    fn calculate_peptide_plane(atom: &BackboneAtoms) -> (Vec3, Vec3) {
        // Direction along strand (C->CA)
        let strand_dir = (atom.ca - atom.c).normalize();

        // Peptide plane normal (using N, CA, C, O to define plane)
        let plane_vec1 = atom.ca - atom.n;
        let plane_vec2 = atom.c - atom.ca;
        let normal = plane_vec1.cross(plane_vec2).normalize();

        (strand_dir, normal)
    }

    /// Generate a mesh around the curve
    fn generate_tube_mesh(curve: &[Vec3], radius: f32, segments: usize) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        // Generate circles around each point
        for (i, &center) in curve.iter().enumerate() {
            let forward = if i < curve.len() - 1 {
                (curve[i + 1] - center).normalize()
            } else {
                (center - curve[i - 1]).normalize()
            };
            let right = if forward.abs_diff_eq(Vec3::Y, 0.01) {
                Vec3::X
            } else {
                forward.cross(Vec3::Y).normalize()
            };
            let up = forward.cross(right);
            // Create vertices around the circle
            for j in 0..segments {
                let angle = (j as f32 / segments as f32) * std::f32::consts::TAU;
                let x = angle.cos();
                let y = angle.sin();
                let pos = center + (right * x + up * y) * radius;
                let normal = (pos - center).normalize();
                positions.push([pos.x, pos.y, pos.z]);
                normals.push([normal.x, normal.y, normal.z]);
                uvs.push([
                    i as f32 / (curve.len() - 1) as f32,
                    j as f32 / segments as f32,
                ]);
            }
        }
        // Generate indices for triangles
        for i in 0..curve.len() - 1 {
            for j in 0..segments {
                let next_j = (j + 1) % segments;
                let current_ring = i * segments;
                let next_ring = (i + 1) * segments;
                indices.push(current_ring + j);
                indices.push(next_ring + j);
                indices.push(current_ring + next_j);
                indices.push(current_ring + next_j);
                indices.push(next_ring + j);
                indices.push(next_ring + next_j);
            }
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(indices.iter().map(|&i| i as u32).collect()));
        mesh
    }

    // A simplified secondary structure detection based on CA geometry
    // In a real implementation, this would use established algorithms like DSSP
    fn detect_secondary_structure(ca_positions: &[Vec3]) -> Vec<SecondaryStructure> {
        let mut sec_structures = vec![SecondaryStructure::Loop; ca_positions.len()];

        // Simplified approach: check distances between i and i+3/i+4 residues
        for i in 0..ca_positions.len() {
            if i + 3 < ca_positions.len() {
                let dist = (ca_positions[i] - ca_positions[i + 3]).length();
                if dist < 6.0 && dist > 4.5 {
                    // Approximate helix criteria
                    sec_structures[i] = SecondaryStructure::Helix;
                }
            }
        }

        // Smooth out single residue assignments
        let mut smoothed = sec_structures.clone();
        for i in 1..sec_structures.len() - 1 {
            if sec_structures[i - 1] == sec_structures[i + 1]
                && sec_structures[i] != sec_structures[i - 1]
            {
                smoothed[i] = sec_structures[i - 1];
            }
        }

        smoothed
    }

    fn generate_cartoon_mesh(curve: &[Vec3], sec_structures: &[SecondaryStructure]) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        // Control how many segments around the tube
        let segments = 16;

        // Generate cross-section profiles for each point in the curve
        for (i, &center) in curve.iter().enumerate() {
            // Get secondary structure type and use it to determine radius and shape
            let sec_type = if i < sec_structures.len() {
                sec_structures[i]
            } else {
                // Interpolate for in-between points
                SecondaryStructure::Loop
            };

            // Calculate tangent direction
            let forward = if i < curve.len() - 1 {
                (curve[i + 1] - center).normalize()
            } else {
                (center - curve[i - 1]).normalize()
            };

            // Calculate normal and binormal for the tube
            let right = if forward.abs_diff_eq(Vec3::Y, 0.01) {
                Vec3::X
            } else {
                forward.cross(Vec3::Y).normalize()
            };
            let up = forward.cross(right);

            // Determine tube profile based on secondary structure
            for j in 0..segments {
                let angle = (j as f32 / segments as f32) * std::f32::consts::TAU;
                let (x, y) = match sec_type {
                    SecondaryStructure::Helix => {
                        // Cylindrical profile for helices
                        let radius = 1.2;
                        (angle.cos() * radius, angle.sin() * radius)
                    }
                    SecondaryStructure::Sheet => {
                        // Flattened profile for sheets
                        let width = 2.0;
                        let height = 0.5;
                        (angle.cos() * width, angle.sin() * height)
                    }
                    SecondaryStructure::Loop => {
                        // Thin tube for loops
                        let radius = 0.6;
                        (angle.cos() * radius, angle.sin() * radius)
                    }
                };

                let pos = center + (right * x + up * y);
                let normal = (pos - center).normalize();

                positions.push([pos.x, pos.y, pos.z]);
                normals.push([normal.x, normal.y, normal.z]);
                uvs.push([i as f32 / curve.len() as f32, j as f32 / segments as f32]);
            }
        }

        // Generate triangle indices
        for i in 0..curve.len() - 1 {
            for j in 0..segments {
                let next_j = (j + 1) % segments;
                let current_ring = i * segments;
                let next_ring = (i + 1) * segments;

                // Two triangles form a quad between consecutive rings
                indices.push(current_ring + j);
                indices.push(next_ring + j);
                indices.push(current_ring + next_j);

                indices.push(current_ring + next_j);
                indices.push(next_ring + j);
                indices.push(next_ring + next_j);
            }
        }

        // Create the final mesh
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(indices.iter().map(|&i| i as u32).collect()));

        mesh
    }

    // Function to generate loop mesh (thin tube connecting structured elements)
    fn generate_loop_mesh(backbone_atoms: &[BackboneAtoms], segment: &[usize]) -> Mesh {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();

        // Parameters for loop
        let tube_radius = 0.2; // Thin tube for loops (Å)
        let tube_segments = 8; // Cross-section circle segments
        let curve_segments = 3; // Segments between residues for smooth curve

        // Extract atoms for this loop segment
        let loop_atoms: Vec<&BackboneAtoms> = segment
            .iter()
            .filter_map(|idx| backbone_atoms.iter().find(|a| a.residue_index == *idx))
            .collect();

        if loop_atoms.len() < 2 {
            return Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        }

        // Create CA positions
        let ca_positions: Vec<Vec3> = loop_atoms.iter().map(|atom| atom.ca).collect();

        // Create a smooth curve through CA atoms
        let curve_points = Structure::create_smooth_curve(&ca_positions, curve_segments);

        // For loops, we simply generate a thin tube following the CA trace
        Structure::generate_tube_mesh(&curve_points, tube_radius, tube_segments)
    }

    fn render_wireframe(&self) -> Mesh {
        todo!()
    }
    fn render_cartoon(&self) -> Mesh {
        // Main implementation
        let backbone_atoms = Structure::extract_backbone_atoms(&self.pdb);
        println!("Backbone atoms found: {}", backbone_atoms.len());

        // Extract CA positions for secondary structure detection
        let ca_positions: Vec<Vec3> = backbone_atoms.iter().map(|atom| atom.ca).collect();
        println!("CA positions extracted: {}", ca_positions.len());

        // Detect secondary structures - returns Vec<SecondaryStructure>
        let secondary_structures = Structure::detect_secondary_structure(&ca_positions);

        println!(
            "Secondary structures detected: {}",
            secondary_structures.len()
        );
        println!(
            "Secondary structure types: {:?}",
            secondary_structures
                .iter()
                .fold([0, 0, 0], |mut counts, &ss| {
                    match ss {
                        SecondaryStructure::Helix => counts[0] += 1,
                        SecondaryStructure::Sheet => counts[1] += 1,
                        SecondaryStructure::Loop => counts[2] += 1,
                    }
                    counts
                })
        );

        // Create combined mesh from all segments
        let mut combined_mesh =
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());

        // First, group the secondary structures into segments of the same type
        let mut segments: Vec<(SecondaryStructure, Vec<usize>)> = Vec::new();
        let mut current_type = secondary_structures[0];
        let mut current_segment = vec![0];

        for i in 1..secondary_structures.len() {
            if secondary_structures[i] == current_type {
                current_segment.push(i);
            } else {
                segments.push((current_type, current_segment.clone()));
                current_type = secondary_structures[i];
                current_segment = vec![i];
            }
        }

        println!("Segments identified: {}", segments.len());

        // Add the last segment
        if !current_segment.is_empty() {
            segments.push((current_type, current_segment));
        }

        // // Now iterate through the segments and generate appropriate meshes
        // for (structure_type, segment) in segments {
        //     let segment_mesh = match structure_type {
        //         SecondaryStructure::Helix => {
        //             Structure::generate_alpha_helix_mesh(&backbone_atoms, &segment)
        //         }
        //         SecondaryStructure::Sheet => {
        //             Structure::generate_beta_sheet_mesh(&backbone_atoms, &segment)
        //         }
        //         SecondaryStructure::Loop => {
        //             Structure::generate_loop_mesh(&backbone_atoms, &segment)
        //         }
        //     };

        //     combined_mesh.merge(&segment_mesh);
        // }

        let mut valid_meshes = Vec::new();

        // Now iterate through the segments and generate appropriate meshes
        for (i, (structure_type, segment)) in segments.iter().enumerate() {
            println!(
                "Processing segment {} of type {:?} with {} residues",
                i,
                structure_type,
                segment.len()
            );
            println!(
                "Segment indices: {:?}",
                &segment[0..std::cmp::min(5, segment.len())]
            );

            // Check if segment indices are valid
            // let valid_indices = segment.iter().all(|&idx| idx < backbone_atoms.len());
            // println!("All segment indices valid: {}", valid_indices);

            let segment_mesh = match structure_type {
                SecondaryStructure::Helix => {
                    Structure::generate_alpha_helix_mesh(&backbone_atoms, segment)
                }
                SecondaryStructure::Sheet => {
                    Structure::generate_beta_sheet_mesh(&backbone_atoms, segment)
                }
                SecondaryStructure::Loop => Structure::generate_loop_mesh(&backbone_atoms, segment),
            };

            println!(
                "After generating segment mesh, vertex count: {}",
                segment_mesh.count_vertices()
            );

            // Only merge if we have vertices
            if segment_mesh.count_vertices() > 0 {
                valid_meshes.push(segment_mesh);
            }
        }

        // If we have any valid meshes, combine them
        if valid_meshes.is_empty() {
            println!("No valid meshes found!");
            // Return an empty mesh since we couldn't generate anything
            return Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        }

        // Start with the first mesh
        let mut combined_mesh = valid_meshes[0].clone();
        println!(
            "Starting with mesh of {} vertices",
            combined_mesh.count_vertices()
        );

        // Merge the rest
        for (i, mesh) in valid_meshes.iter().enumerate().skip(1) {
            println!("Merging mesh {} with {} vertices", i, mesh.count_vertices());
            let before_count = combined_mesh.count_vertices();
            combined_mesh.merge(mesh);
            let after_count = combined_mesh.count_vertices();
            println!(
                "After merge: {} vertices (added {})",
                after_count,
                after_count - before_count
            );
        }

        println!(
            "Final combined mesh vertices: {}",
            combined_mesh.count_vertices()
        );
        combined_mesh
    }

    fn render_ballandstick(&self) -> Mesh {
        let radius = 0.5;
        let mut combined_mesh = self
            .pdb
            .iter_coords_and_elements()
            .map(|(coord, element_str)| {
                let center = Vec3::new(coord[0], coord[1], coord[2]);
                let mut sphere_mesh = Sphere::new(radius).mesh().build();
                let vertex_count = sphere_mesh.count_vertices();
                let color = self.color_scheme.get_color(element_str).to_srgba();
                let color_array =
                    vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vertex_count];
                sphere_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_array);
                sphere_mesh = sphere_mesh.translated_by(center);
                sphere_mesh.compute_smooth_normals();
                sphere_mesh
            })
            .reduce(|mut acc, mesh| {
                acc.merge(&mesh);
                acc
            })
            .unwrap();

        // Add bond cylinders using iterators
        if let Some(bonds) = self.pdb.get_bonds() {
            let coords = self.pdb.get_coords();
            bonds
                .iter()
                .filter_map(|bond| {
                    let (atom1, atom2) = bond.get_atom_indices();
                    let pos1 = Vec3::from_array(*coords.get(atom1 as usize)?);
                    let pos2 = Vec3::from_array(*coords.get(atom2 as usize)?);

                    // Calculate cylinder properties
                    let center = (pos1 + pos2) / 2.0;
                    let direction = pos2 - pos1;
                    let height = direction.length();
                    let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize());

                    // Create and transform cylinder mesh
                    let mut cylinder_mesh = Cylinder {
                        radius: 0.5,
                        half_height: height / 2.0, // Note: we divide height by 2 since it expects half_height
                    }
                    .mesh()
                    .build();

                    // Apply transformation
                    cylinder_mesh = cylinder_mesh.transformed_by(Transform {
                        translation: center,
                        rotation,
                        ..default()
                    });

                    // Add colors
                    let cylinder_vertex_count = cylinder_mesh.count_vertices();
                    let cylinder_colors =
                        vec![Vec4::new(0.5, 0.5, 0.5, 0.5); cylinder_vertex_count];
                    cylinder_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, cylinder_colors);
                    Some(cylinder_mesh)
                })
                .for_each(|cylinder_mesh| {
                    combined_mesh.merge(&cylinder_mesh);
                });
        } else {
            println!("No-Bonds found!!")
        }
        combined_mesh
    }
    /// Internal fn for rendering spheres.
    fn render_spheres(&self) -> Mesh {
        self.pdb
            .iter_coords_and_elements()
            .map(|(coord, element)| {
                let center = Vec3::new(coord[0], coord[1], coord[2]);
                let radius = element
                    .atomic_radius()
                    .van_der_waals
                    .expect("Van der waals not defined") as f32;
                let mut sphere_mesh = Sphere::new(radius).mesh().build();
                let vertex_count = sphere_mesh.count_vertices();
                let color = self.color_scheme.get_color(element).to_srgba();
                let color_array =
                    vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vertex_count];
                sphere_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_array);
                sphere_mesh = sphere_mesh.translated_by(center);
                sphere_mesh.compute_smooth_normals();
                sphere_mesh
            })
            .reduce(|mut acc, mesh| {
                acc.merge(&mesh);
                acc
            })
            .unwrap()
    }

    fn render_putty(&self) -> Result<Mesh, Error> {
        let c_alphas: Vec<Vec3> = self
            .pdb
            .iter_residues_aminoacid()
            .map(|residue| {
                let ca = residue.find_atom_by_name("CA").expect("CA in all residues");
                Vec3::from_array(*ca.coords())
            })
            .collect();
        let curve = Structure::create_smooth_curve(&c_alphas, 3);
        let tube_mesh = Structure::generate_tube_mesh(&curve, 0.3, 16);
        Ok(tube_mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferritin_core::load_structure;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_pdb_to_mesh() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let ac = load_structure(molfile)?;
        let structure = Structure::builder().pdb(ac).build();
        assert_eq!(structure.pdb.get_size(), 2154);
        let mesh = structure.to_mesh();
        assert_eq!(mesh.count_vertices(), 779748);
        Ok(())
    }

    #[test]
    fn test_pdb_to_mesh_cartoon() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let ac = load_structure(molfile)?;
        let structure = Structure::builder()
            .pdb(ac)
            .rendertype(RenderOptions::Cartoon)
            .build();

        assert_eq!(structure.pdb.get_size(), 2154);
        let mesh = structure.to_mesh();
        assert!(mesh.count_vertices() > 0);
        Ok(())
    }
}
