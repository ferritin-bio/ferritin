//! Structure.
//!
//! Struct for rendering with Bevy
//!
//!

use super::ColorScheme;
use crate::selection::AtomMask;
use bevy::asset::RenderAssetUsages;
use bevy::math::Vec4;
use bevy::prelude::{
    Color, Component, Cylinder, Mesh, MeshBuilder, Meshable, Quat, Sphere, StandardMaterial,
    Transform, Vec3, default,
};
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bon::Builder;
use ferritin_core::Model;

/// Enum representing various rendering options.
///
/// Each of these enums represents a rendering path that can be used by a `Structure`
///
/// Down the Line: allow passing an arbitrary function that maps PDB to mesh.
///
#[derive(Clone, PartialEq)]
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

    /// Returns `(Mesh, atom_map)` where `atom_map[vertex_idx]` is the source atom index.
    /// Applicable to Solid, BallAndStick, and Wireframe render types.
    /// `mask` filters which atoms contribute to the mesh; `None` includes all atoms.
    pub fn to_mesh_with_atom_map(&self, mask: Option<&AtomMask>) -> (Mesh, Vec<usize>) {
        match &self.rendertype {
            RenderOptions::Solid => self.render_spheres_mapped(mask),
            RenderOptions::BallAndStick => self.render_ballandstick_mapped(mask),
            RenderOptions::Wireframe => self.render_wireframe_mapped(mask),
            _ => (self.to_mesh(), Vec::new()),
        }
    }

    /// Returns `(Mesh, residue_map)` where `residue_map[vertex_idx]` is the 0-based residue
    /// iteration index that produced that vertex. Applicable to Cartoon and Putty.
    /// `mask` filters which residues contribute; `None` includes all residues.
    pub fn to_mesh_with_residue_map(&self, mask: Option<&AtomMask>) -> (Mesh, Vec<usize>) {
        match &self.rendertype {
            RenderOptions::Cartoon => self.render_cartoon_mapped(mask),
            RenderOptions::Putty => self.render_putty_mapped(mask),
            _ => (self.to_mesh(), Vec::new()),
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
        if count == 0 {
            Vec3::ZERO
        } else {
            sum / count as f32
        }
    }

    /// Maximum CA-CA distance (Å) treated as a continuous backbone. A trans
    /// peptide bond gives ~3.8 Å; anything beyond this is a chain break or a
    /// stretch of missing residues, across which the ribbon must not spline.
    const CA_BREAK_DIST: f32 = 5.0;

    /// BallAndStick geometry (Å). The stick is half the ball radius so the rep
    /// reads as spheres joined by thin sticks rather than uniform fat tubes
    /// (ferritin-c1k.1).
    const BALL_RADIUS: f32 = 0.3;
    const STICK_RADIUS: f32 = 0.15;

    /// Wireframe/Line bond radius (Å). A GPU `LineList` rasterizes to a hard 1px
    /// edge with no normals, which aliases badly on large structures and can't be
    /// lit or anti-aliased. Rendering the Line representation as thin triangle
    /// cylinders instead gives it real geometry (smooth-shaded, MSAA-covered)
    /// while staying visually distinct from BallAndStick's thicker sticks
    /// (ferritin-t0h.9).
    const LINE_RADIUS: f32 = 0.13;

    /// Builds the two half-length cylinders for a ball-and-stick bond, split at
    /// the bond's midpoint so the caller can color/map each half to its own
    /// endpoint atom — real bonds in molecular viewers are two-toned, half per
    /// atom, rather than the whole stick taking on a single atom's color
    /// (ferritin-c1k.2). Returns `None` for a degenerate (zero-length) bond.
    fn half_bond_cylinders(pos1: Vec3, pos2: Vec3, radius: f32) -> Option<(Mesh, Mesh)> {
        let direction = pos2 - pos1;
        let height = direction.length();
        if height < 1e-6 {
            return None;
        }
        let mid = (pos1 + pos2) * 0.5;
        let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize());
        let half_height = height / 4.0;
        let make_half = |center: Vec3| -> Mesh {
            Cylinder { radius, half_height }
                .mesh()
                .build()
                .transformed_by(Transform {
                    translation: center,
                    rotation,
                    ..default()
                })
        };
        let first = make_half((pos1 + mid) * 0.5);
        let second = make_half((mid + pos2) * 0.5);
        Some((first, second))
    }

    /// Whether two consecutive backbone residues should start a new ribbon
    /// segment rather than be splined together. A break is any of: a gap in the
    /// (masked) residue-iteration index, a chain change, or a CA-CA distance
    /// beyond a single peptide bond (missing residues / HETATM insertion).
    /// See ferritin-t0h.1.
    fn is_backbone_break(
        prev_idx: usize,
        prev_chain: &str,
        prev_ca: Vec3,
        idx: usize,
        chain: &str,
        ca: Vec3,
    ) -> bool {
        idx != prev_idx + 1 || chain != prev_chain || ca.distance(prev_ca) > Self::CA_BREAK_DIST
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

    /// Rotation-minimizing frames along a curve (parallel transport). Returns one
    /// orthonormal `(right, up)` pair per curve point, both perpendicular to the
    /// local tangent. The pair rotates *minimally* from one point to the next, so
    /// the extruded tube cross-section never flips 180° at a kink the way the old
    /// `tangent × global_Y` frame did — that flip produced the white starburst
    /// spikes at joints (ferritin-t0h.2).
    fn sweep_frames(curve: &[Vec3]) -> Vec<(Vec3, Vec3)> {
        let n = curve.len();
        let mut frames = Vec::with_capacity(n);
        if n == 0 {
            return frames;
        }

        let tangent = |i: usize| -> Vec3 {
            let raw = if i + 1 < n {
                curve[i + 1] - curve[i]
            } else {
                curve[i] - curve[i - 1]
            };
            raw.normalize_or_zero()
        };

        let t0 = {
            let t = tangent(0);
            if t.length_squared() > 1e-8 {
                t
            } else {
                Vec3::Z
            }
        };
        let seed = if t0.abs_diff_eq(Vec3::Y, 0.01) {
            Vec3::X
        } else {
            Vec3::Y
        };
        let mut right = t0.cross(seed).normalize();
        let mut up = t0.cross(right).normalize();
        frames.push((right, up));

        for i in 1..n {
            let t_prev = tangent(i - 1);
            let t_cur = tangent(i);
            let axis = t_prev.cross(t_cur);
            let sin_a = axis.length();
            if sin_a > 1e-6 {
                let axis_n = axis / sin_a;
                let cos_a = t_prev.dot(t_cur).clamp(-1.0, 1.0);
                let angle = sin_a.atan2(cos_a);
                right = Quat::from_axis_angle(axis_n, angle) * right;
            }
            // Re-orthonormalize against the current tangent to kill drift.
            right = (right - t_cur * right.dot(t_cur)).normalize_or_zero();
            if right.length_squared() < 1e-8 {
                let fallback = if t_cur.abs_diff_eq(Vec3::Y, 0.01) {
                    Vec3::X
                } else {
                    Vec3::Y
                };
                right = t_cur.cross(fallback).normalize();
            }
            up = t_cur.cross(right).normalize();
            frames.push((right, up));
        }

        frames
    }

    /// Per-residue reference directions for orienting the cartoon ribbon's flat
    /// cross-section, using the Carson-Bugg "virtual torsion" construction:
    /// normal(i) = normalize((CA(i) - CA(i-1)) x (CA(i+1) - CA(i))). Unlike a
    /// pure parallel-transported frame (whose orientation is arbitrary — seeded
    /// from a global axis with no relation to the actual backbone geometry),
    /// this direction is derived from the real CA trace, so consecutive
    /// residues' flat faces consistently track the backbone's own twist instead
    /// of an arbitrary seed (ferritin-t0h.3).
    ///
    /// This construction flips sign roughly every residue in extended (beta)
    /// conformations because of the backbone's zigzag; consecutive directions
    /// are sign-corrected (flipped if their dot product is negative) so the
    /// ribbon doesn't whip 180° every residue. Endpoints reuse their nearest
    /// interior neighbor. One direction per input CA position.
    fn peptide_plane_normals(ca_positions: &[Vec3]) -> Vec<Vec3> {
        let n = ca_positions.len();
        let mut normals = vec![Vec3::Y; n];
        if n < 3 {
            return normals;
        }
        for i in 1..n - 1 {
            let b1 = ca_positions[i] - ca_positions[i - 1];
            let b2 = ca_positions[i + 1] - ca_positions[i];
            let normal = b1.cross(b2).normalize_or_zero();
            normals[i] = if normal.length_squared() > 1e-8 {
                normal
            } else {
                Vec3::Y
            };
        }
        normals[0] = normals[1];
        normals[n - 1] = normals[n - 2];
        for i in 1..n {
            if normals[i].dot(normals[i - 1]) < 0.0 {
                normals[i] = -normals[i];
            }
        }
        normals
    }

    /// Normalized linear interpolation between two directions; falls back to `a`
    /// if the blend degenerates (e.g. `a` and `b` nearly cancel).
    fn nlerp_direction(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        let blended = a * (1.0 - t) + b * t;
        let n = blended.normalize_or_zero();
        if n.length_squared() > 1e-8 { n } else { a }
    }

    /// Like [`sweep_frames`], but steers each ring's `up` vector toward an
    /// explicit per-point hint (projected perpendicular to the local tangent)
    /// instead of purely parallel-transporting an arbitrary seed. Falls back to
    /// parallel transport from the previous frame when the hint is degenerate
    /// (nearly parallel to the tangent), and sign-corrects against the previous
    /// frame so the ribbon doesn't flip 180° between rings. Used to orient the
    /// cartoon ribbon with the real backbone geometry (ferritin-t0h.3) rather
    /// than an arbitrary global-axis seed.
    fn sweep_frames_oriented(curve: &[Vec3], target_up: &[Vec3]) -> Vec<(Vec3, Vec3)> {
        let n = curve.len();
        let mut frames = Vec::with_capacity(n);
        if n == 0 {
            return frames;
        }

        let tangent = |i: usize| -> Vec3 {
            let raw = if i + 1 < n {
                curve[i + 1] - curve[i]
            } else {
                curve[i] - curve[i - 1]
            };
            let t = raw.normalize_or_zero();
            if t.length_squared() > 1e-8 { t } else { Vec3::Z }
        };

        let mut prev_up: Option<Vec3> = None;
        for i in 0..n {
            let t_cur = tangent(i);
            let hint = target_up.get(i).copied().unwrap_or(Vec3::Y);
            let mut up = (hint - t_cur * hint.dot(t_cur)).normalize_or_zero();
            if up.length_squared() < 1e-8 {
                // Hint nearly parallel to the tangent (degenerate projection):
                // fall back to parallel-transporting the previous frame's up.
                up = match prev_up {
                    Some(p) => (p - t_cur * p.dot(t_cur)).normalize_or_zero(),
                    None => {
                        let seed = if t_cur.abs_diff_eq(Vec3::Y, 0.01) {
                            Vec3::X
                        } else {
                            Vec3::Y
                        };
                        t_cur.cross(seed).normalize_or_zero()
                    }
                };
            }
            if let Some(p) = prev_up {
                if up.dot(p) < 0.0 {
                    up = -up;
                }
            }
            let right = up.cross(t_cur).normalize_or_zero();
            // Re-derive up from (tangent, right) so the pair is exactly
            // orthonormal (matches the `up = tangent x right` convention used
            // by the mesh-generation cross-sections).
            let up = t_cur.cross(right).normalize_or_zero();
            frames.push((right, up));
            prev_up = Some(up);
        }

        frames
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

    /// Generate a mesh around a curve path, with one radius per curve point (used
    /// by Putty's synthetic thickness placeholder — see `synthetic_putty_radii`).
    ///
    /// # Panics
    /// Panics if `radii.len() != curve.len()`.
    fn generate_tube_mesh_varying(curve: &[Vec3], radii: &[f32], segments: usize) -> Mesh {
        assert_eq!(radii.len(), curve.len(), "one radius per curve point");
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        // Rotation-minimizing frames avoid cross-section flips at kinks (ferritin-t0h.2).
        let frames = Structure::sweep_frames(curve);
        // Generate circles around each point
        for (i, &center) in curve.iter().enumerate() {
            let radius = radii[i];
            let (right, up) = frames[i];
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
            } else if (-170.0..=-50.0).contains(&phi_d) && (psi_d >= 60.0 || psi_d <= -150.0) {
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

    /// Generate a cartoon ribbon mesh from a smooth CA-trace curve, per-point
    /// secondary structure assignments, and per-point ribbon-orientation hints.
    ///
    /// Cross-sections: helix → circle r=1.2, sheet → ellipse 2.0×0.5, loop → circle r=0.4.
    /// Vertex colors: helix=salmon, sheet=periwinkle, loop=light-green.
    fn generate_cartoon_mesh(
        curve: &[Vec3],
        sec_structures: &[SecondaryStructure],
        target_up: &[Vec3],
    ) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut colors: Vec<Vec4> = Vec::new();
        let mut indices = Vec::new();

        // Control how many segments around the tube
        let segments = 16;

        // Orient the ribbon with the real backbone geometry (ferritin-t0h.3)
        // instead of an arbitrary parallel-transported seed, while still
        // avoiding cross-section flips at kinks (ferritin-t0h.2).
        let frames = Structure::sweep_frames_oriented(curve, target_up);

        // Generate cross-section profiles for each point in the curve
        for (i, &center) in curve.iter().enumerate() {
            let sec_type = if i < sec_structures.len() {
                sec_structures[i]
            } else {
                SecondaryStructure::Loop
            };

            let (right, up) = frames[i];

            let color = match sec_type {
                SecondaryStructure::Helix => Vec4::new(1.0, 0.6, 0.6, 1.0),
                SecondaryStructure::Sheet => Vec4::new(0.6, 0.6, 1.0, 1.0),
                SecondaryStructure::Loop => Vec4::new(0.6, 0.9, 0.6, 1.0),
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
        // Collect (chain_id, ca_position) for all amino-acid residues, then walk
        // consecutive pairs emitting a thin cylinder segment for each edge whose
        // endpoints share a chain (for a chain A→B→C→D: [A,B], [B,C], [C,D]).
        let ca_by_chain: Vec<(String, Vec3)> = self
            .pdb
            .residues_aminoacid()
            .filter_map(|residue| {
                residue.find_atom_by_name("CA").map(|ca| {
                    (
                        residue.chain_id().to_string(),
                        Vec3::from_array(ca.coords()),
                    )
                })
            })
            .collect();

        let mut combined: Option<Mesh> = None;
        for window in ca_by_chain.windows(2) {
            let (chain_a, pos_a) = &window[0];
            let (chain_b, pos_b) = &window[1];
            if chain_a == chain_b {
                if let Some((mut first, second)) =
                    Structure::half_bond_cylinders(*pos_a, *pos_b, Structure::LINE_RADIUS)
                {
                    let _ = first.merge(&second);
                    match &mut combined {
                        None => combined = Some(first),
                        Some(acc) => {
                            let _ = acc.merge(&first);
                        }
                    }
                }
            }
        }

        combined.unwrap_or_else(|| Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all()))
    }

    fn render_cartoon(&self) -> Mesh {
        // Delegate to the mapped path (mask = None) so the ribbon is split into
        // contiguous backbone segments at chain breaks / missing-residue gaps
        // rather than splining straight through them (ferritin-t0h.1).
        self.render_cartoon_mapped(None).0
    }

    fn render_ballandstick(&self) -> Mesh {
        let radius = Structure::BALL_RADIUS;
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

        // Add bond cylinders, split at the midpoint and colored per endpoint
        // atom so a bond visibly transitions between its two atoms' element
        // colors instead of taking on a single flat color (ferritin-c1k.2).
        let elements: Vec<ferritin_core::info::elements::Element> =
            self.pdb.atoms().map(|a| a.element()).collect();
        let bonds = &self.pdb.hierarchy.bonds;
        if !bonds.atom_a.is_empty() {
            (0..bonds.atom_a.len())
                .filter_map(|i| {
                    let idx_a = bonds.atom_a[i] as usize;
                    let idx_b = bonds.atom_b[i] as usize;
                    let pos1 = Vec3::from_array(self.pdb.coord(idx_a));
                    let pos2 = Vec3::from_array(self.pdb.coord(idx_b));
                    let (mut first, mut second) =
                        Structure::half_bond_cylinders(pos1, pos2, Structure::STICK_RADIUS)?;

                    let color_a = self.color_scheme.get_color(&elements[idx_a]).to_srgba();
                    let vc1 = first.count_vertices();
                    first.insert_attribute(
                        Mesh::ATTRIBUTE_COLOR,
                        vec![Vec4::new(color_a.red, color_a.green, color_a.blue, color_a.alpha); vc1],
                    );

                    let color_b = self.color_scheme.get_color(&elements[idx_b]).to_srgba();
                    let vc2 = second.count_vertices();
                    second.insert_attribute(
                        Mesh::ATTRIBUTE_COLOR,
                        vec![Vec4::new(color_b.red, color_b.green, color_b.blue, color_b.alpha); vc2],
                    );

                    let _ = first.merge(&second);
                    Some(first)
                })
                .for_each(|bond_mesh| {
                    let _ = combined_mesh.merge(&bond_mesh);
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
        // Delegate to the mapped path (mask = None) so the tube is split at chain
        // breaks / missing-residue gaps rather than splining through them
        // (ferritin-t0h.1).
        self.render_putty_mapped(None).0
    }

    /// Placeholder tube-radius profile for Putty, until per-atom b_factor is
    /// plumbed through from Model into AtomCollection (tracked in ferritin-clv).
    /// Without it, Putty rendered as a constant-radius tube identical to
    /// Cartoon's loop tube (ferritin-ala.10). A smooth sinusoidal variation along
    /// the chain at least signals "this thickness carries data" rather than
    /// looking like an unstyled Cartoon; it is not meaningful per-residue data.
    fn synthetic_putty_radii(n: usize) -> Vec<f32> {
        const BASE_RADIUS: f32 = 0.3;
        const AMPLITUDE: f32 = 0.22;
        const CYCLES: f32 = 4.0;
        (0..n)
            .map(|i| {
                let t = i as f32 / n.max(1) as f32;
                BASE_RADIUS + AMPLITUDE * (t * std::f32::consts::TAU * CYCLES).sin()
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Masked / mapped variants
    // -----------------------------------------------------------------------

    fn render_spheres_mapped(&self, mask: Option<&AtomMask>) -> (Mesh, Vec<usize>) {
        let mut atom_map: Vec<usize> = Vec::new();
        let mut combined: Option<Mesh> = None;

        for atom in self.pdb.atoms() {
            let idx = atom.index();
            if let Some(m) = mask {
                if !m.0[idx] {
                    continue;
                }
            }

            let coord = atom.coords();
            let center = Vec3::new(coord[0], coord[1], coord[2]);
            let element = atom.element();
            let radius = element.atomic_radius().van_der_waals.expect("vdw radius") as f32;
            let mut sphere_mesh = Sphere::new(radius).mesh().build();
            let vc = sphere_mesh.count_vertices();
            let color = self.color_scheme.get_color(&element).to_srgba();
            sphere_mesh.insert_attribute(
                Mesh::ATTRIBUTE_COLOR,
                vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vc],
            );
            sphere_mesh = sphere_mesh.translated_by(center);
            sphere_mesh.compute_smooth_normals();
            atom_map.extend(std::iter::repeat(idx).take(vc));
            match &mut combined {
                None => combined = Some(sphere_mesh),
                Some(acc) => {
                    let _ = acc.merge(&sphere_mesh);
                }
            }
        }

        let mesh = combined.unwrap_or_else(|| {
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        });
        (mesh, atom_map)
    }

    fn render_wireframe_mapped(&self, mask: Option<&AtomMask>) -> (Mesh, Vec<usize>) {
        let mut atom_map: Vec<usize> = Vec::new();
        let mut combined: Option<Mesh> = None;

        // Collect all CA atoms (retain ALL so window() neighbours stay correct).
        let ca_data: Vec<(String, usize, Vec3)> = self
            .pdb
            .residues_aminoacid()
            .filter_map(|residue| {
                residue.find_atom_by_name("CA").map(|ca| {
                    let idx = ca.index();
                    (
                        residue.chain_id().to_string(),
                        idx,
                        Vec3::from_array(ca.coords()),
                    )
                })
            })
            .collect();

        // Emit an edge only when same chain AND both CA atoms pass the mask. Each
        // edge is a pair of half-length cylinders, one per endpoint atom, so the
        // atom_map (and downstream per-vertex coloring) stays split at the
        // midpoint exactly like BallAndStick bonds (ferritin-c1k.2).
        for window in ca_data.windows(2) {
            let (chain_a, idx_a, pos_a) = &window[0];
            let (chain_b, idx_b, pos_b) = &window[1];
            let a_ok = mask.map_or(true, |m| m.0[*idx_a]);
            let b_ok = mask.map_or(true, |m| m.0[*idx_b]);
            if chain_a == chain_b && a_ok && b_ok {
                if let Some((first, second)) =
                    Structure::half_bond_cylinders(*pos_a, *pos_b, Structure::LINE_RADIUS)
                {
                    atom_map.extend(std::iter::repeat(*idx_a).take(first.count_vertices()));
                    atom_map.extend(std::iter::repeat(*idx_b).take(second.count_vertices()));
                    let mut seg = first;
                    let _ = seg.merge(&second);
                    match &mut combined {
                        None => combined = Some(seg),
                        Some(acc) => {
                            let _ = acc.merge(&seg);
                        }
                    }
                }
            }
        }

        let mesh = combined.unwrap_or_else(|| {
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        });
        (mesh, atom_map)
    }

    fn render_ballandstick_mapped(&self, mask: Option<&AtomMask>) -> (Mesh, Vec<usize>) {
        let radius = Structure::BALL_RADIUS;
        let mut atom_map: Vec<usize> = Vec::new();
        let mut combined: Option<Mesh> = None;

        // Atom spheres (filtered by mask).
        for atom in self.pdb.atoms() {
            let idx = atom.index();
            if let Some(m) = mask {
                if !m.0[idx] {
                    continue;
                }
            }

            let coord = atom.coords();
            let center = Vec3::new(coord[0], coord[1], coord[2]);
            let element = atom.element();
            let mut sphere_mesh = Sphere::new(radius).mesh().build();
            let vc = sphere_mesh.count_vertices();
            let color = self.color_scheme.get_color(&element).to_srgba();
            sphere_mesh.insert_attribute(
                Mesh::ATTRIBUTE_COLOR,
                vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vc],
            );
            sphere_mesh = sphere_mesh.translated_by(center);
            sphere_mesh.compute_smooth_normals();
            atom_map.extend(std::iter::repeat(idx).take(vc));
            match &mut combined {
                None => combined = Some(sphere_mesh),
                Some(acc) => {
                    let _ = acc.merge(&sphere_mesh);
                }
            }
        }

        // Bond cylinders: skip if either endpoint is masked out. Split at the
        // midpoint so each half maps to its own endpoint atom — the MVS color
        // theme recolors every vertex by its atom_map entry, so without this
        // split the whole stick took on a single atom's color instead of
        // visibly transitioning between the two (ferritin-c1k.2).
        let bonds = &self.pdb.hierarchy.bonds;
        if !bonds.atom_a.is_empty() {
            for i in 0..bonds.atom_a.len() {
                let idx_a = bonds.atom_a[i] as usize;
                let idx_b = bonds.atom_b[i] as usize;
                if let Some(m) = mask {
                    if !m.0[idx_a] || !m.0[idx_b] {
                        continue;
                    }
                }
                let pos1 = Vec3::from_array(self.pdb.coord(idx_a));
                let pos2 = Vec3::from_array(self.pdb.coord(idx_b));
                let Some((mut first, mut second)) =
                    Structure::half_bond_cylinders(pos1, pos2, Structure::STICK_RADIUS)
                else {
                    continue;
                };

                let vc1 = first.count_vertices();
                first.insert_attribute(
                    Mesh::ATTRIBUTE_COLOR,
                    vec![Vec4::new(0.5, 0.5, 0.5, 0.5); vc1],
                );
                atom_map.extend(std::iter::repeat(idx_a).take(vc1));

                let vc2 = second.count_vertices();
                second.insert_attribute(
                    Mesh::ATTRIBUTE_COLOR,
                    vec![Vec4::new(0.5, 0.5, 0.5, 0.5); vc2],
                );
                atom_map.extend(std::iter::repeat(idx_b).take(vc2));

                let _ = first.merge(&second);
                match &mut combined {
                    None => combined = Some(first),
                    Some(acc) => {
                        let _ = acc.merge(&first);
                    }
                }
            }
        }

        let mesh = combined.unwrap_or_else(|| {
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        });
        (mesh, atom_map)
    }

    fn render_cartoon_mapped(&self, mask: Option<&AtomMask>) -> (Mesh, Vec<usize>) {
        let segments_per_residue: usize = 4;
        let cross_segments: usize = 16;

        // Collect (residue_iter_idx, chain_id, BackboneAtoms) for residues whose CA passes the mask.
        let indexed_backbone: Vec<(usize, String, BackboneAtoms)> = self
            .pdb
            .residues_aminoacid()
            .enumerate()
            .filter_map(|(res_iter_idx, residue)| {
                let ca = residue.find_atom_by_name("CA")?;
                if let Some(m) = mask {
                    if !m.0[ca.index()] {
                        return None;
                    }
                }
                let n = residue.find_atom_by_name("N")?;
                let c = residue.find_atom_by_name("C")?;
                // O is required so the residue counts as a complete backbone unit,
                // matching the previous behaviour, even though only N/CA/C feed the
                // dihedral-based secondary-structure classifier.
                let _o = residue.find_atom_by_name("O")?;
                Some((
                    res_iter_idx,
                    residue.chain_id().to_string(),
                    BackboneAtoms {
                        ca: Vec3::from_array(ca.coords()),
                        n: Vec3::from_array(n.coords()),
                        c: Vec3::from_array(c.coords()),
                    },
                ))
            })
            .collect();

        if indexed_backbone.is_empty() {
            return (
                Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all()),
                Vec::new(),
            );
        }

        // Split into contiguous backbone segments so the spline never connects
        // across a ribbon break. A break is any of: a gap in the masked residue
        // sequence, a chain change, or a CA-CA jump larger than a peptide bond
        // (missing residues / HETATM insertion). See ferritin-t0h.1.
        let mut segments: Vec<Vec<(usize, BackboneAtoms)>> = Vec::new();
        let mut current: Vec<(usize, BackboneAtoms)> = Vec::new();
        let mut last_chain: Option<String> = None;
        for (idx, chain, bb) in indexed_backbone {
            if let (Some(last), Some(prev_chain)) = (current.last(), last_chain.as_deref()) {
                if Structure::is_backbone_break(last.0, prev_chain, last.1.ca, idx, &chain, bb.ca) {
                    segments.push(std::mem::take(&mut current));
                }
            }
            last_chain = Some(chain);
            current.push((idx, bb));
        }
        if !current.is_empty() {
            segments.push(current);
        }

        let mut residue_map: Vec<usize> = Vec::new();
        let mut combined: Option<Mesh> = None;

        for segment in segments {
            if segment.len() < 2 {
                continue;
            }
            let (res_indices, backbone_atoms): (Vec<usize>, Vec<BackboneAtoms>) =
                segment.into_iter().unzip();
            let secondary_structures = Structure::detect_secondary_structure(&backbone_atoms);
            let ca_positions: Vec<Vec3> = backbone_atoms.iter().map(|a| a.ca).collect();
            let curve = Structure::create_smooth_curve(&ca_positions, segments_per_residue);
            let curve_sec: Vec<SecondaryStructure> = (0..curve.len())
                .map(|i| {
                    secondary_structures
                        .get(i / segments_per_residue)
                        .copied()
                        .unwrap_or(SecondaryStructure::Loop)
                })
                .collect();
            // Per-residue backbone-derived ribbon orientation (ferritin-t0h.3),
            // interpolated onto the finer curve so the flat cross-section twists
            // smoothly with the real geometry instead of an arbitrary seed.
            let residue_normals = Structure::peptide_plane_normals(&ca_positions);
            let curve_up: Vec<Vec3> = (0..curve.len())
                .map(|i| {
                    let res_idx = i / segments_per_residue;
                    let t = (i % segments_per_residue) as f32 / segments_per_residue as f32;
                    let n0 = residue_normals[res_idx.min(residue_normals.len() - 1)];
                    let n1 = residue_normals[(res_idx + 1).min(residue_normals.len() - 1)];
                    Structure::nlerp_direction(n0, n1, t)
                })
                .collect();
            let seg_mesh = Structure::generate_cartoon_mesh(&curve, &curve_sec, &curve_up);
            let seg_verts = seg_mesh.count_vertices();
            // Map each vertex back to its residue iteration index.
            for v in 0..seg_verts {
                let curve_pt = v / cross_segments;
                let res_in_seg = (curve_pt / segments_per_residue).min(res_indices.len() - 1);
                residue_map.push(res_indices[res_in_seg]);
            }
            match &mut combined {
                None => combined = Some(seg_mesh),
                Some(acc) => {
                    let _ = acc.merge(&seg_mesh);
                }
            }
        }

        let mesh = combined.unwrap_or_else(|| {
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        });
        (mesh, residue_map)
    }

    fn render_putty_mapped(&self, mask: Option<&AtomMask>) -> (Mesh, Vec<usize>) {
        const CROSS_SEGMENTS: usize = 16;
        const CURVE_SEGMENTS: usize = 3;

        // Collect (res_iter_idx, chain_id, CA position) for unmasked residues.
        let ca_data: Vec<(usize, String, Vec3)> = self
            .pdb
            .residues_aminoacid()
            .enumerate()
            .filter_map(|(res_iter_idx, residue)| {
                let ca = residue.find_atom_by_name("CA")?;
                if let Some(m) = mask {
                    if !m.0[ca.index()] {
                        return None;
                    }
                }
                Some((
                    res_iter_idx,
                    residue.chain_id().to_string(),
                    Vec3::from_array(ca.coords()),
                ))
            })
            .collect();

        if ca_data.len() < 2 {
            return (
                Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all()),
                Vec::new(),
            );
        }

        // Split into contiguous segments at chain breaks / missing-residue gaps
        // so the tube never splines straight through them (ferritin-t0h.1).
        let mut segments: Vec<Vec<(usize, Vec3)>> = Vec::new();
        let mut current: Vec<(usize, Vec3)> = Vec::new();
        let mut last_chain: Option<String> = None;
        for (idx, chain, ca) in ca_data {
            if let (Some(last), Some(prev_chain)) = (current.last(), last_chain.as_deref()) {
                if Structure::is_backbone_break(last.0, prev_chain, last.1, idx, &chain, ca) {
                    segments.push(std::mem::take(&mut current));
                }
            }
            last_chain = Some(chain);
            current.push((idx, ca));
        }
        if !current.is_empty() {
            segments.push(current);
        }

        let mut residue_map: Vec<usize> = Vec::new();
        let mut combined: Option<Mesh> = None;

        for segment in segments {
            if segment.len() < 2 {
                continue;
            }
            let (res_indices, ca_positions): (Vec<usize>, Vec<Vec3>) = segment.into_iter().unzip();
            let curve = Structure::create_smooth_curve(&ca_positions, CURVE_SEGMENTS);
            let radii = Structure::synthetic_putty_radii(curve.len());
            let seg_mesh = Structure::generate_tube_mesh_varying(&curve, &radii, CROSS_SEGMENTS);
            let seg_verts = seg_mesh.count_vertices();
            for v in 0..seg_verts {
                let curve_pt = v / CROSS_SEGMENTS;
                let res_in_list = (curve_pt / CURVE_SEGMENTS).min(res_indices.len() - 1);
                residue_map.push(res_indices[res_in_list]);
            }
            match &mut combined {
                None => combined = Some(seg_mesh),
                Some(acc) => {
                    let _ = acc.merge(&seg_mesh);
                }
            }
        }

        let mesh = combined.unwrap_or_else(|| {
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        });
        (mesh, residue_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::{AtomMask, evaluate_selector};
    use ferritin_core::load_model;
    use ferritin_molviewspec::molviewspec::nodes::{ComponentSelector, ComponentSelectorT};
    use ferritin_test_data::TestFile;

    // ferritin-c1k.1: BallAndStick balls must read as thicker than the sticks
    // joining them, not as one uniform-radius licorice tube.
    #[test]
    fn test_ballandstick_stick_thinner_than_ball() {
        assert!(
            Structure::STICK_RADIUS < Structure::BALL_RADIUS,
            "stick radius {} must be smaller than ball radius {}",
            Structure::STICK_RADIUS,
            Structure::BALL_RADIUS
        );
    }

    // ferritin-c1k.2: a bond must be split at its midpoint so each half can be
    // colored/mapped to its own endpoint atom, instead of the whole stick
    // taking on a single atom's color.

    #[test]
    fn test_half_bond_cylinders_splits_at_midpoint() {
        let pos1 = Vec3::new(0.0, 0.0, 0.0);
        let pos2 = Vec3::new(0.0, 4.0, 0.0);
        let (first, second) = Structure::half_bond_cylinders(pos1, pos2, 0.1)
            .expect("a valid (non-degenerate) bond must produce two halves");

        let y_range = |mesh: &Mesh| -> (f32, f32) {
            let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
                bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v,
                _ => panic!("unexpected position format"),
            };
            positions.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                (lo.min(p[1]), hi.max(p[1]))
            })
        };

        let (first_min, first_max) = y_range(&first);
        let (second_min, second_max) = y_range(&second);

        assert!(
            first_min >= -1e-3 && first_max <= 2.0 + 1e-3,
            "first half should span [0, midpoint=2.0], got [{first_min}, {first_max}]"
        );
        assert!(
            second_min >= 2.0 - 1e-3 && second_max <= 4.0 + 1e-3,
            "second half should span [midpoint=2.0, 4], got [{second_min}, {second_max}]"
        );
    }

    #[test]
    fn test_half_bond_cylinders_degenerate_returns_none() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert!(Structure::half_bond_cylinders(p, p, 0.1).is_none());
    }

    #[test]
    fn test_ballandstick_bond_second_endpoint_gets_own_atom_map_entries() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_01().create_temp()?;
        let model = load_model(molfile).unwrap();

        // Find an atom that is only ever a bond's *second* endpoint (atom_b),
        // never its first (atom_a). Under the old bug, every bond's vertices
        // mapped only to atom_a, so this atom's atom_map count would be exactly
        // its own sphere's vertex count with zero contribution from bonds.
        let atom_a: std::collections::HashSet<usize> = model
            .hierarchy
            .bonds
            .atom_a
            .iter()
            .map(|&x| x as usize)
            .collect();
        let target_idx = model
            .hierarchy
            .bonds
            .atom_b
            .iter()
            .map(|&x| x as usize)
            .find(|idx| !atom_a.contains(idx))
            .expect("expected an atom that is only ever a bond's second endpoint");

        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::BallAndStick)
            .build();
        let (mesh, atom_map) = structure.to_mesh_with_atom_map(None);
        assert_eq!(mesh.count_vertices(), atom_map.len());

        let sphere_vc = Sphere::new(Structure::BALL_RADIUS).mesh().build().count_vertices();
        let count_in_map = atom_map.iter().filter(|&&i| i == target_idx).count();
        assert!(
            count_in_map > sphere_vc,
            "atom {target_idx} is only ever a bond's second endpoint; its atom_map \
             count ({count_in_map}) must exceed its own sphere's vertex count \
             ({sphere_vc}), proving a bond's second half maps to it instead of \
             every bond vertex mapping only to the first endpoint"
        );
        Ok(())
    }

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

    // ferritin-t0h.1: cartoon/putty must not spline across chain breaks.

    #[test]
    fn test_backbone_break_detection() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.8, 0.0, 0.0); // one peptide bond away
        // Contiguous, same chain, adjacent CAs → no break.
        assert!(!Structure::is_backbone_break(0, "A", a, 1, "A", b));
        // Chain change → break, even if spatially close.
        assert!(Structure::is_backbone_break(0, "A", a, 1, "B", b));
        // Residue-index gap (e.g. masked-out residue) → break.
        assert!(Structure::is_backbone_break(0, "A", a, 2, "A", b));
        // Large CA-CA jump (missing residues) → break.
        let far = Vec3::new(20.0, 0.0, 0.0);
        assert!(Structure::is_backbone_break(0, "A", a, 1, "A", far));
    }

    #[test]
    fn test_sweep_frames_are_continuous_and_orthonormal() {
        // A curve that kinks sharply and passes near the global Y axis — exactly
        // the case where the old tangent×Y frame flipped and produced spikes.
        let curve = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.2, 0.0),
            Vec3::new(1.5, 1.5, 0.0), // turns to head up +Y
            Vec3::new(1.6, 3.0, 0.2),
            Vec3::new(1.0, 4.0, 1.0),
            Vec3::new(0.0, 4.2, 2.0),
        ];
        let frames = Structure::sweep_frames(&curve);
        assert_eq!(frames.len(), curve.len());
        for (i, (right, up)) in frames.iter().enumerate() {
            assert!((right.length() - 1.0).abs() < 1e-3, "right[{i}] not unit");
            assert!((up.length() - 1.0).abs() < 1e-3, "up[{i}] not unit");
            assert!(right.dot(*up).abs() < 1e-3, "right·up[{i}] not orthogonal");
        }
        // Consecutive frames must not flip: the minimal-rotation frame keeps the
        // "right" vector pointing broadly the same way step to step (dot > 0).
        for i in 1..frames.len() {
            assert!(
                frames[i].0.dot(frames[i - 1].0) > 0.0,
                "frame right vector flipped between {} and {i}",
                i - 1
            );
        }
    }

    // ferritin-t0h.3: ribbon orientation should track real backbone geometry
    // (peptide-plane normal), not an arbitrary parallel-transported seed.

    #[test]
    fn test_peptide_plane_normals_sign_corrected_and_unit() {
        // A zigzag CA trace mimicking an extended (beta-strand) backbone: the
        // raw Carson-Bugg cross product flips sign roughly every residue here,
        // which is exactly what the sign-correction pass must undo.
        let ca_positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 1.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(5.0, 1.0, 0.0),
        ];
        let normals = Structure::peptide_plane_normals(&ca_positions);
        assert_eq!(normals.len(), ca_positions.len());
        for (i, n) in normals.iter().enumerate() {
            assert!((n.length() - 1.0).abs() < 1e-3, "normals[{i}] not unit");
        }
        for i in 1..normals.len() {
            assert!(
                normals[i].dot(normals[i - 1]) >= 0.0,
                "peptide-plane normal flipped between residue {} and {i}",
                i - 1
            );
        }
    }

    #[test]
    fn test_sweep_frames_oriented_tracks_hint_and_stays_continuous() {
        // Straight curve along +Z; hint consistently points toward +X, so `up`
        // (perpendicular to the tangent) should end up parallel to +X at every
        // point, not just whatever a parallel-transported seed happened to pick.
        let curve = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 3.0),
        ];
        let hints = vec![Vec3::X; curve.len()];
        let frames = Structure::sweep_frames_oriented(&curve, &hints);
        assert_eq!(frames.len(), curve.len());
        for (i, (right, up)) in frames.iter().enumerate() {
            assert!((right.length() - 1.0).abs() < 1e-3, "right[{i}] not unit");
            assert!((up.length() - 1.0).abs() < 1e-3, "up[{i}] not unit");
            assert!(right.dot(*up).abs() < 1e-3, "right·up[{i}] not orthogonal");
            assert!(
                up.dot(Vec3::X).abs() > 0.99,
                "up[{i}]={up:?} should track the +X hint, not an arbitrary seed"
            );
        }
    }

    #[test]
    fn test_sweep_frames_oriented_falls_back_when_hint_degenerate() {
        // Hint parallel to the tangent (straight +Z curve, hint also +Z) can't
        // be projected to a perpendicular direction; must fall back to a valid
        // orthonormal frame instead of producing NaN/zero vectors.
        let curve = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        let hints = vec![Vec3::Z; curve.len()];
        let frames = Structure::sweep_frames_oriented(&curve, &hints);
        for (i, (right, up)) in frames.iter().enumerate() {
            assert!(right.is_finite() && up.is_finite(), "frame[{i}] not finite");
            assert!((right.length() - 1.0).abs() < 1e-3, "right[{i}] not unit");
            assert!((up.length() - 1.0).abs() < 1e-3, "up[{i}] not unit");
            assert!(right.dot(*up).abs() < 1e-3, "right·up[{i}] not orthogonal");
        }
    }

    #[test]
    fn test_cartoon_segments_multi_chain_4hhb() -> anyhow::Result<()> {
        // 4hhb has 4 protein chains; the cartoon must render as several
        // disconnected ribbons, not one curve splined through all of them.
        let (molfile, _handle) = TestFile::mvs_4hhb().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Cartoon)
            .build();

        let (mesh, residue_map) = structure.render_cartoon_mapped(None);
        assert!(mesh.count_vertices() > 0, "cartoon mesh must not be empty");
        assert_eq!(mesh.count_vertices(), residue_map.len());

        // A single unsegmented curve through all N amino-acid residues would
        // produce (N-1)*segments_per_residue ring centers. Each additional
        // segment removes one inter-residue span, so a genuinely split ribbon
        // has strictly fewer ring centers than the unsegmented curve.
        let n_res = structure.pdb.residues_aminoacid().count();
        let cross_segments = 16;
        let segments_per_residue = 4;
        let unsegmented_rings = (n_res - 1) * segments_per_residue;
        let actual_rings = mesh.count_vertices() / cross_segments;
        assert!(
            actual_rings < unsegmented_rings,
            "expected segmentation to drop ring centers: {actual_rings} vs {unsegmented_rings}"
        );
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

        let vertex_count = mesh.count_vertices();
        assert!(vertex_count > 0, "wireframe mesh has no vertices");
        // Line rep is rendered as thin triangle cylinders, not a GPU LineList,
        // so it gets real normals and doesn't alias to 1px (ferritin-t0h.9).
        assert_eq!(mesh.primitive_topology(), PrimitiveTopology::TriangleList);
        assert!(
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some(),
            "wireframe mesh should carry normals for lighting"
        );

        Ok(())
    }

    // -----------------------------------------------------------------------
    // T2-25 to T2-27: Solid masked rendering
    // -----------------------------------------------------------------------

    #[test]
    fn test_solid_none_mask_regression() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Solid)
            .build();
        let (mesh_none, map_none) = structure.to_mesh_with_atom_map(None);
        assert!(
            mesh_none.count_vertices() > 0,
            "None mask must produce vertices"
        );
        assert_eq!(
            mesh_none.count_vertices(),
            map_none.len(),
            "atom_map len must equal vertex count"
        );
        Ok(())
    }

    #[test]
    fn test_solid_protein_mask_fewer_verts() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let ac = ferritin_core::AtomCollection::from(&model);
        let prot_mask = evaluate_selector(
            &ComponentSelector::Selector(ComponentSelectorT::Protein),
            &ac,
        );

        // Build a Model from the original (not filtered) for rendering.
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Solid)
            .build();
        let (mesh_all, _) = structure.to_mesh_with_atom_map(None);
        let (mesh_prot, map_prot) = structure.to_mesh_with_atom_map(Some(&prot_mask));

        assert!(
            mesh_prot.count_vertices() > 0,
            "protein mask must still produce vertices"
        );
        assert!(
            mesh_prot.count_vertices() <= mesh_all.count_vertices(),
            "masked mesh must have <= vertices than full mesh"
        );
        assert_eq!(
            mesh_prot.count_vertices(),
            map_prot.len(),
            "atom_map len must equal vertex count for protein mask"
        );
        Ok(())
    }

    #[test]
    fn test_solid_all_false_mask_empty() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let n = model.n_atoms();
        let empty_mask = AtomMask(vec![false; n]);
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Solid)
            .build();
        let (mesh, map) = structure.to_mesh_with_atom_map(Some(&empty_mask));
        assert_eq!(
            mesh.count_vertices(),
            0,
            "all-false mask must produce 0 vertices"
        );
        assert_eq!(map.len(), 0);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // T2-28 to T2-30: Wireframe masked rendering
    // -----------------------------------------------------------------------

    #[test]
    fn test_wireframe_none_mask_atom_map_consistency() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Wireframe)
            .build();
        let (mesh, map) = structure.to_mesh_with_atom_map(None);
        assert_eq!(
            mesh.count_vertices(),
            map.len(),
            "wireframe: atom_map len must equal vertex count"
        );
        assert_eq!(
            mesh.count_vertices() % 2,
            0,
            "wireframe: vertex count must be even"
        );
        Ok(())
    }

    #[test]
    fn test_wireframe_protein_mask_fewer_edges() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let ac = ferritin_core::AtomCollection::from(&model);
        let prot_mask = evaluate_selector(
            &ComponentSelector::Selector(ComponentSelectorT::Protein),
            &ac,
        );
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Wireframe)
            .build();
        let (mesh_all, _) = structure.to_mesh_with_atom_map(None);
        let (mesh_prot, map_prot) = structure.to_mesh_with_atom_map(Some(&prot_mask));
        assert!(
            mesh_prot.count_vertices() <= mesh_all.count_vertices(),
            "masked wireframe must have <= edges"
        );
        assert_eq!(mesh_prot.count_vertices(), map_prot.len());
        Ok(())
    }

    #[test]
    fn test_wireframe_all_false_empty() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let empty_mask = AtomMask(vec![false; model.n_atoms()]);
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Wireframe)
            .build();
        let (mesh, map) = structure.to_mesh_with_atom_map(Some(&empty_mask));
        assert_eq!(mesh.count_vertices(), 0);
        assert_eq!(map.len(), 0);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // T2-31 to T2-33: Cartoon residue map
    // -----------------------------------------------------------------------

    #[test]
    fn test_cartoon_none_mask_residue_map() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Cartoon)
            .build();
        let (mesh, residue_map) = structure.to_mesh_with_residue_map(None);
        assert!(mesh.count_vertices() > 0);
        assert_eq!(
            mesh.count_vertices(),
            residue_map.len(),
            "cartoon: residue_map len must equal vertex count"
        );
        Ok(())
    }

    #[test]
    fn test_cartoon_protein_mask_fewer_verts() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let ac = ferritin_core::AtomCollection::from(&model);
        let prot_mask = evaluate_selector(
            &ComponentSelector::Selector(ComponentSelectorT::Protein),
            &ac,
        );
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Cartoon)
            .build();
        let (mesh_all, _) = structure.to_mesh_with_residue_map(None);
        let (mesh_prot, map_prot) = structure.to_mesh_with_residue_map(Some(&prot_mask));
        assert!(mesh_prot.count_vertices() <= mesh_all.count_vertices());
        assert_eq!(mesh_prot.count_vertices(), map_prot.len());
        Ok(())
    }

    #[test]
    fn test_cartoon_all_false_mask_empty() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let empty_mask = AtomMask(vec![false; model.n_atoms()]);
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Cartoon)
            .build();
        let (mesh, map) = structure.to_mesh_with_residue_map(Some(&empty_mask));
        assert_eq!(mesh.count_vertices(), 0);
        assert_eq!(map.len(), 0);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // T2-34 to T2-40: atom_map and residue_map invariants
    // -----------------------------------------------------------------------

    #[test]
    fn test_atom_map_all_indices_valid() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let n = model.n_atoms();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Solid)
            .build();
        let (_, atom_map) = structure.to_mesh_with_atom_map(None);
        for &idx in &atom_map {
            assert!(
                idx < n,
                "atom_map index {} out of bounds (n_atoms={})",
                idx,
                n
            );
        }
        Ok(())
    }

    #[test]
    fn test_wireframe_atom_map_all_indices_valid() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let n = model.n_atoms();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Wireframe)
            .build();
        let (_, atom_map) = structure.to_mesh_with_atom_map(None);
        for &idx in &atom_map {
            assert!(idx < n, "wireframe atom_map index {} out of bounds", idx);
        }
        Ok(())
    }

    #[test]
    fn test_residue_map_values_non_decreasing() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Cartoon)
            .build();
        let (_, residue_map) = structure.to_mesh_with_residue_map(None);
        for w in residue_map.windows(2) {
            assert!(
                w[0] <= w[1],
                "residue_map must be non-decreasing but got {} then {}",
                w[0],
                w[1]
            );
        }
        Ok(())
    }

    #[test]
    fn test_putty_residue_map_length() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Putty)
            .build();
        let (mesh, residue_map) = structure.to_mesh_with_residue_map(None);
        assert_eq!(
            mesh.count_vertices(),
            residue_map.len(),
            "putty: residue_map len must equal vertex count"
        );
        Ok(())
    }

    // ferritin-ala.10: Putty rendered as a constant-radius tube, pixel-identical
    // to Cartoon's loop tube, since AtomCollection has no per-atom b_factor yet.
    // Until that lands (ferritin-clv), the tube radius should at least vary
    // along the chain as a visual placeholder.
    #[test]
    fn test_synthetic_putty_radii_vary() {
        let radii = Structure::synthetic_putty_radii(40);
        let min = radii.iter().cloned().fold(f32::MAX, f32::min);
        let max = radii.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            max - min > 0.1,
            "putty radii should vary noticeably along the chain, got range {min}..{max}"
        );
        assert!(
            radii.iter().all(|&r| r > 0.0),
            "all synthetic putty radii must stay positive"
        );
    }

    #[test]
    fn test_putty_mesh_radius_varies_along_chain() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Putty)
            .build();
        let mesh = structure.to_mesh();
        let positions = match mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .expect("putty mesh must have positions")
        {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v,
            _ => panic!("unexpected position format"),
        };
        // Cross-section rings are 16 vertices each (CROSS_SEGMENTS in render_putty);
        // compare each ring's average radial distance from its own centroid.
        let ring_size = 16;
        let ring_radius = |ring: &[[f32; 3]]| -> f32 {
            let centroid = ring
                .iter()
                .fold(Vec3::ZERO, |acc, p| acc + Vec3::from_array(*p))
                / ring.len() as f32;
            ring.iter()
                .map(|p| (Vec3::from_array(*p) - centroid).length())
                .sum::<f32>()
                / ring.len() as f32
        };
        let ring_radii: Vec<f32> = positions
            .chunks(ring_size)
            .filter(|chunk| chunk.len() == ring_size)
            .map(ring_radius)
            .collect();
        let min = ring_radii.iter().cloned().fold(f32::MAX, f32::min);
        let max = ring_radii.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            max - min > 0.1,
            "putty mesh cross-section radius should vary along the chain, got range {min}..{max}"
        );
        Ok(())
    }

    #[test]
    fn test_masked_solid_atom_map_indices_in_mask() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_04().create_temp()?;
        let model = load_model(molfile).unwrap();
        let ac = ferritin_core::AtomCollection::from(&model);
        let prot_mask = evaluate_selector(
            &ComponentSelector::Selector(ComponentSelectorT::Protein),
            &ac,
        );
        let structure = Structure::builder()
            .pdb(model)
            .rendertype(RenderOptions::Solid)
            .build();
        let (_, atom_map) = structure.to_mesh_with_atom_map(Some(&prot_mask));
        for &idx in &atom_map {
            assert!(
                prot_mask.0[idx],
                "atom_map must only contain indices that pass the mask"
            );
        }
        Ok(())
    }
}
