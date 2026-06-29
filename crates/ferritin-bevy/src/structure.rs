//! Structure.
//!
//! Struct for rendering with Bevy
//!
//!

use super::ColorScheme;
use bevy::math::Vec4;
use bevy::prelude::{
    Color, Component, Cylinder, Mesh, MeshBuilder, Meshable, Quat, Sphere, StandardMaterial,
    Transform, Vec3, default,
};
use bevy::asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bon::Builder;
use ferritin_core::Model;

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
    ca: Vec3,
    n: Vec3,
    c: Vec3,
    o: Vec3,
    residue_index: usize,
}

/// Define Everything Needed to render
#[derive(Builder, Component)]
pub struct Structure {
    pdb: Model,
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
            RenderOptions::Putty => self.render_putty(),
        }
    }

    /// Returns the material for this structure.
    ///
    /// Bevy 0.19 auto-enables vertex colors when the mesh contains ATTRIBUTE_COLOR,
    /// so no material flag is needed here.
    pub fn get_material(&self) -> StandardMaterial {
        self.material.clone()
    }

    /// Compute the centroid (mean position) of all atoms.
    pub fn centroid(&self) -> Vec3 {
        let mut sum = Vec3::ZERO;
        let mut count = 0usize;
        for atom in self.pdb.atoms() {
            sum += Vec3::from_array(atom.coords());
            count += 1;
        }
        if count == 0 { Vec3::ZERO } else { sum / count as f32 }
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

    fn extract_backbone_atoms(pdb: &Model) -> Vec<BackboneAtoms> {
        let mut backbone_data = Vec::new();

        for residue in pdb.residues_aminoacid() {
            if let (Some(ca), Some(n), Some(c), Some(o)) = (
                residue.find_atom_by_name("CA"),
                residue.find_atom_by_name("N"),
                residue.find_atom_by_name("C"),
                residue.find_atom_by_name("O"),
            ) {
                backbone_data.push(BackboneAtoms {
                    ca: Vec3::from_array(ca.coords()),
                    n: Vec3::from_array(n.coords()),
                    c: Vec3::from_array(c.coords()),
                    o: Vec3::from_array(o.coords()),
                    residue_index: residue.residue_id() as usize,
                });
            }
        }

        backbone_data
    }

    /// Generate a mesh around a curve path (used by Putty).
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

    /// Torsion angle (radians) between the planes defined by four sequential atoms.
    /// Uses the atan2 formulation so the result is in (-π, π].
    fn dihedral(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> f32 {
        let b1 = p2 - p1;
        let b2 = p3 - p2;
        let b3 = p4 - p3;
        let n1 = b1.cross(b2);
        let n2 = b2.cross(b3);
        let m1 = n1.cross(b2.normalize());
        f32::atan2(m1.dot(n2), n1.dot(n2))
    }

    /// Secondary structure classification via backbone phi/psi dihedral angles.
    ///
    /// Phi(i)  = dihedral( C(i-1), N(i),  CA(i), C(i)  )
    /// Psi(i)  = dihedral( N(i),   CA(i), C(i),  N(i+1) )
    ///
    /// Thresholds (degrees, converted to radians internally):
    ///   Helix: phi ∈ [-90, -30], psi ∈ [-80, 0]
    ///   Sheet: phi ∈ [-170, -50], psi ∈ [60, 180] or psi ∈ [-180, -150]
    fn detect_secondary_structure(atoms: &[BackboneAtoms]) -> Vec<SecondaryStructure> {
        let n = atoms.len();
        let mut sec = vec![SecondaryStructure::Loop; n];

        let to_deg = |r: f32| r.to_degrees();

        for i in 1..n.saturating_sub(1) {
            let phi = Self::dihedral(atoms[i - 1].c, atoms[i].n, atoms[i].ca, atoms[i].c);
            let psi = Self::dihedral(atoms[i].n, atoms[i].ca, atoms[i].c, atoms[i + 1].n);
            let phi_d = to_deg(phi);
            let psi_d = to_deg(psi);

            if (-90.0..=-30.0).contains(&phi_d) && (-80.0..=0.0).contains(&psi_d) {
                sec[i] = SecondaryStructure::Helix;
            } else if (-170.0..=-50.0).contains(&phi_d)
                && (psi_d >= 60.0 || psi_d <= -150.0)
            {
                sec[i] = SecondaryStructure::Sheet;
            }
        }

        // Smooth out single-residue islands
        let orig = sec.clone();
        for i in 1..n - 1 {
            if orig[i - 1] == orig[i + 1] && orig[i] != orig[i - 1] {
                sec[i] = orig[i - 1];
            }
        }

        sec
    }

    /// Generate a cartoon ribbon mesh from a smooth CA-trace curve and per-point
    /// secondary structure assignments.
    ///
    /// Cross-sections: helix → circle r=1.2, sheet → ellipse 2.0×0.5, loop → circle r=0.4.
    /// Vertex colors: helix=salmon, sheet=periwinkle, loop=light-green.
    fn generate_cartoon_mesh(curve: &[Vec3], sec_structures: &[SecondaryStructure]) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut colors: Vec<Vec4> = Vec::new();
        let mut indices = Vec::new();

        // Control how many segments around the tube
        let segments = 16;

        // Generate cross-section profiles for each point in the curve
        for (i, &center) in curve.iter().enumerate() {
            let sec_type = if i < sec_structures.len() {
                sec_structures[i]
            } else {
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

            let color = match sec_type {
                SecondaryStructure::Helix => Vec4::new(1.0, 0.6, 0.6, 1.0),
                SecondaryStructure::Sheet => Vec4::new(0.6, 0.6, 1.0, 1.0),
                SecondaryStructure::Loop  => Vec4::new(0.6, 0.9, 0.6, 1.0),
            };

            // Determine tube profile based on secondary structure
            for j in 0..segments {
                let angle = (j as f32 / segments as f32) * std::f32::consts::TAU;
                let (x, y) = match sec_type {
                    SecondaryStructure::Helix => {
                        let radius = 1.2;
                        (angle.cos() * radius, angle.sin() * radius)
                    }
                    SecondaryStructure::Sheet => {
                        // Flattened ribbon profile
                        (angle.cos() * 2.0, angle.sin() * 0.5)
                    }
                    SecondaryStructure::Loop => {
                        let radius = 0.4;
                        (angle.cos() * radius, angle.sin() * radius)
                    }
                };

                let pos = center + (right * x + up * y);
                let normal = (pos - center).normalize();

                positions.push([pos.x, pos.y, pos.z]);
                normals.push([normal.x, normal.y, normal.z]);
                uvs.push([i as f32 / curve.len() as f32, j as f32 / segments as f32]);
                colors.push(color);
            }
        }

        // Generate triangle indices
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
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        mesh.insert_indices(Indices::U32(indices.iter().map(|&i| i as u32).collect()));

        mesh
    }

    fn render_wireframe(&self) -> Mesh {
        // Collect Cα positions grouped by chain.
        // Emit LineList pairs: for a chain A→B→C→D emit [A,B, B,C, C,D].
        let mut positions: Vec<[f32; 3]> = Vec::new();

        // Collect (chain_id, ca_position) for all amino-acid residues
        let ca_by_chain: Vec<(String, Vec3)> = self
            .pdb
            .residues_aminoacid()
            .filter_map(|residue| {
                residue.find_atom_by_name("CA").map(|ca| {
                    (residue.chain_id().to_string(), Vec3::from_array(ca.coords()))
                })
            })
            .collect();

        // Walk consecutive pairs; emit an edge only when both endpoints share a chain.
        for window in ca_by_chain.windows(2) {
            let (chain_a, pos_a) = &window[0];
            let (chain_b, pos_b) = &window[1];
            if chain_a == chain_b {
                positions.push([pos_a.x, pos_a.y, pos_a.z]);
                positions.push([pos_b.x, pos_b.y, pos_b.z]);
            }
        }

        let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::all());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh
    }

    fn render_cartoon(&self) -> Mesh {
        let backbone_atoms = Structure::extract_backbone_atoms(&self.pdb);
        if backbone_atoms.len() < 2 {
            return Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
        }

        let secondary_structures = Structure::detect_secondary_structure(&backbone_atoms);
        let ca_positions: Vec<Vec3> = backbone_atoms.iter().map(|a| a.ca).collect();

        // Build a smooth Catmull-Rom curve through the CA trace.
        let segments_per_residue = 4;
        let curve = Structure::create_smooth_curve(&ca_positions, segments_per_residue);

        // Map per-residue secondary structure onto the finer curve points.
        let curve_sec: Vec<SecondaryStructure> = (0..curve.len())
            .map(|i| {
                secondary_structures
                    .get(i / segments_per_residue)
                    .copied()
                    .unwrap_or(SecondaryStructure::Loop)
            })
            .collect();

        Structure::generate_cartoon_mesh(&curve, &curve_sec)
    }

    fn render_ballandstick(&self) -> Mesh {
        let radius = 0.5;
        let mut combined_mesh = self
            .pdb
            .atoms()
            .map(|atom| {
                let coord = atom.coords();
                let center = Vec3::new(coord[0], coord[1], coord[2]);
                let element = atom.element();
                let mut sphere_mesh = Sphere::new(radius).mesh().build();
                let vertex_count = sphere_mesh.count_vertices();
                let color = self.color_scheme.get_color(&element).to_srgba();
                let color_array =
                    vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vertex_count];
                sphere_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_array);
                sphere_mesh = sphere_mesh.translated_by(center);
                sphere_mesh.compute_smooth_normals();
                sphere_mesh
            })
            .reduce(|mut acc, mesh| {
                let _ = acc.merge(&mesh);
                acc
            })
            .unwrap();

        // Add bond cylinders
        let bonds = &self.pdb.hierarchy.bonds;
        if !bonds.atom_a.is_empty() {
            (0..bonds.atom_a.len())
                .filter_map(|i| {
                    let pos1 = Vec3::from_array(self.pdb.coord(bonds.atom_a[i] as usize));
                    let pos2 = Vec3::from_array(self.pdb.coord(bonds.atom_b[i] as usize));
                    let center = (pos1 + pos2) / 2.0;
                    let direction = pos2 - pos1;
                    let height = direction.length();
                    if height < 1e-6 {
                        return None;
                    }
                    let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize());
                    let mut cylinder_mesh = Cylinder {
                        radius: 0.5,
                        half_height: height / 2.0,
                    }
                    .mesh()
                    .build();
                    cylinder_mesh = cylinder_mesh.transformed_by(Transform {
                        translation: center,
                        rotation,
                        ..default()
                    });
                    let cylinder_vertex_count = cylinder_mesh.count_vertices();
                    let cylinder_colors =
                        vec![Vec4::new(0.5, 0.5, 0.5, 0.5); cylinder_vertex_count];
                    cylinder_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, cylinder_colors);
                    Some(cylinder_mesh)
                })
                .for_each(|cylinder_mesh| {
                    let _ = combined_mesh.merge(&cylinder_mesh);
                });
        }
        combined_mesh
    }

    fn render_spheres(&self) -> Mesh {
        self.pdb
            .atoms()
            .map(|atom| {
                let coord = atom.coords();
                let center = Vec3::new(coord[0], coord[1], coord[2]);
                let element = atom.element();
                let radius = element
                    .atomic_radius()
                    .van_der_waals
                    .expect("Van der waals not defined") as f32;
                let mut sphere_mesh = Sphere::new(radius).mesh().build();
                let vertex_count = sphere_mesh.count_vertices();
                let color = self.color_scheme.get_color(&element).to_srgba();
                let color_array =
                    vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vertex_count];
                sphere_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_array);
                sphere_mesh = sphere_mesh.translated_by(center);
                sphere_mesh.compute_smooth_normals();
                sphere_mesh
            })
            .reduce(|mut acc, mesh| {
                let _ = acc.merge(&mesh);
                acc
            })
            .unwrap()
    }

    fn render_putty(&self) -> Mesh {
        let c_alphas: Vec<Vec3> = self
            .pdb
            .residues_aminoacid()
            .map(|residue| {
                let ca = residue.find_atom_by_name("CA").expect("CA in all residues");
                Vec3::from_array(ca.coords())
            })
            .collect();
        let curve = Structure::create_smooth_curve(&c_alphas, 3);
        Structure::generate_tube_mesh(&curve, 0.3, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferritin_core::load_model;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_pdb_to_mesh() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder().pdb(model).build();
        assert_eq!(structure.pdb.n_atoms(), 2154);
        let _mesh = structure.to_mesh();
        // assert_eq!(_mesh.count_vertices(), 779748);
        Ok(())
    }

    #[test]
    fn test_pdb_to_mesh_cartoon() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Cartoon)
            .build();

        assert_eq!(structure.pdb.n_atoms(), 2154);
        let mesh = structure.to_mesh();
        assert!(mesh.count_vertices() > 0);
        Ok(())
    }

    #[test]
    fn test_render_wireframe() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Wireframe)
            .build();

        let mesh = structure.to_mesh();

        // (1) There must be vertices
        let vertex_count = mesh.count_vertices();
        assert!(vertex_count > 0, "wireframe mesh has no vertices");

        // (2) LineList topology requires pairs — vertex count must be even
        assert_eq!(
            vertex_count % 2,
            0,
            "wireframe vertex count {vertex_count} is not even (LineList requires pairs)"
        );

        // (3) Topology must be LineList
        assert_eq!(
            mesh.primitive_topology(),
            PrimitiveTopology::LineList,
            "expected LineList topology"
        );

        Ok(())
    }
}
