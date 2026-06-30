//! MVS Execution Engine.
//!
//! Walks a parsed MolViewSpec [`State`] tree and spawns Bevy entities that render
//! it. The engine is *fault tolerant*: an error in one branch of the tree emits an
//! [`MvsError`] message and execution continues with sibling nodes, so a single bad
//! download or empty selector never aborts the whole scene.
//!
//! # Entry point
//! Send a [`LoadMvsEvent`] message; [`execute_mvs_on_load`] (registered by
//! [`MvsPlugin`]) parses the state, clears any previously spawned [`MvsEntity`]
//! entities, and rebuilds the scene.
//!
//! # MVP limitations
//! - HTTP/HTTPS downloads are not supported (emit [`MvsError::UnsupportedNode`]);
//!   use local file paths only.
//! - Assembly / symmetry structures render the plain asymmetric unit and emit an
//!   [`MvsError::UnsupportedNode`] warning.
//! - `component_from_uri` / `*_from_uri` / `*_from_source` nodes are skipped with an
//!   [`MvsError::UnsupportedNode`].
//! - Surface representation degrades to VdW spheres (Solid).
//! - When multiple `focus` nodes exist, the last one processed (document order) wins.

use std::path::PathBuf;

use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use ferritin_core::{AtomCollection, Model, load_model};
use ferritin_molviewspec::molviewspec::nodes::{
    CameraParams, CanvasParams, ColorT, ComponentInlineParams, ComponentSelector,
    FocusInlineParams, KindT, LabelInlineParams, LineParams, Node, NodeParams, ParseFormatT,
    RepresentationParams, RepresentationTypeT, SphereParams, State, StructureParams,
    StructureTypeT, TransformParams,
};

use crate::colors::{VertexMapKind, apply_mvs_colors, color_t_to_bevy};
use crate::selection::{AtomMask, evaluate_selector};
use crate::structure::{RenderOptions, Structure};

// ---------------------------------------------------------------------------
// Public ECS types
// ---------------------------------------------------------------------------

/// Marker attached to *every* entity spawned by the executor. On each reload all
/// `MvsEntity` entities are despawned before the new state is built.
#[derive(Component)]
pub struct MvsEntity;

/// A MVS `label` node, spawned at the centroid of its component's atoms. Headless
/// builds carry only the text + [`Transform`]; a UI layer turns it into billboard text.
#[derive(Component)]
pub struct MvsLabel(pub String);

/// Holds the most recently loaded [`State`] (for UI tree inspection). Replaced on
/// each successful load.
#[derive(Resource, Default)]
pub struct MvsStateResource(pub Option<State>);

/// Orbit-camera target state written by `camera` and `focus` nodes. A viewer reads
/// this resource to position its camera; `focus` always wins over `camera` because
/// it is applied after the root-level camera node.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub up: Vec3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        let radius = 50.0_f32.hypot(100.0);
        Self {
            focus: Vec3::ZERO,
            radius,
            yaw: 0.0,
            pitch: (50.0 / radius).asin(),
            up: Vec3::Y,
        }
    }
}

/// Request to load a MolViewSpec state.
#[derive(Message, Clone, Debug)]
pub enum LoadMvsEvent {
    /// Path to a `.mvsj` document on the local filesystem.
    FromFile(PathBuf),
    /// A `.mvsj` document already in memory.
    FromString(String),
}

/// Non-fatal problems encountered while executing a state. Each is emitted as a
/// message; a status bar can subscribe to surface them to the user.
#[derive(Message, Clone, Debug, PartialEq)]
pub enum MvsError {
    /// A structure file referenced by a `download` node was not found / failed to load.
    FileNotFound(PathBuf),
    /// The `.mvsj` document itself failed to parse.
    ParseError(String),
    /// A component's selector matched no atoms; the component was skipped.
    NoAtomsSelected { selector_description: String },
    /// A `transform` node carried a malformed rotation matrix.
    InvalidTransform,
    /// A node kind / feature is not supported in this MVP.
    UnsupportedNode { kind: KindT, reason: String },
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Registers the MVS executor: messages, resources, and the load system.
///
/// Adds the minimal asset infrastructure (`AssetPlugin`, `Mesh` + `StandardMaterial`
/// assets, `ClearColor`) when it is not already present, so the plugin works under
/// both `MinimalPlugins` (headless tests) and `DefaultPlugins`.
pub struct MvsPlugin;

impl Plugin for MvsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<AssetPlugin>() {
            app.add_plugins(AssetPlugin::default());
        }
        if !app.world().contains_resource::<Assets<Mesh>>() {
            app.init_asset::<Mesh>();
        }
        if !app.world().contains_resource::<Assets<StandardMaterial>>() {
            app.init_asset::<StandardMaterial>();
        }
        if !app.world().contains_resource::<ClearColor>() {
            app.insert_resource(ClearColor::default());
        }
        app.init_resource::<MvsStateResource>()
            .init_resource::<OrbitCamera>()
            .add_message::<LoadMvsEvent>()
            .add_message::<MvsError>()
            .add_systems(Update, execute_mvs_on_load);
    }
}

// ---------------------------------------------------------------------------
// Load system
// ---------------------------------------------------------------------------

/// Bundles the mutable world handles the recursive executor needs, plus the error
/// sink. Errors are buffered here and flushed to the [`MvsError`] message channel
/// once traversal completes (keeps `MessageWriter` out of the recursion).
struct Ctx<'a, 'w, 's> {
    commands: &'a mut Commands<'w, 's>,
    meshes: &'a mut Assets<Mesh>,
    materials: &'a mut Assets<StandardMaterial>,
    clear: &'a mut ClearColor,
    orbit: &'a mut OrbitCamera,
    errors: Vec<MvsError>,
}

/// Reacts to [`LoadMvsEvent`]: parse, clear the previous scene, execute the new tree.
#[allow(clippy::too_many_arguments)]
fn execute_mvs_on_load(
    mut load_events: MessageReader<LoadMvsEvent>,
    mut error_writer: MessageWriter<MvsError>,
    existing: Query<Entity, With<MvsEntity>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut clear: ResMut<ClearColor>,
    mut orbit: ResMut<OrbitCamera>,
    mut state_res: ResMut<MvsStateResource>,
) {
    for event in load_events.read() {
        // Parse first; a parse failure aborts only this event.
        let state = match parse_state(event) {
            Ok(s) => s,
            Err(e) => {
                error_writer.write(e);
                continue;
            }
        };

        // Clear the previous scene.
        for entity in existing.iter() {
            commands.entity(entity).despawn();
        }

        let mut ctx = Ctx {
            commands: &mut commands,
            meshes: &mut meshes,
            materials: &mut materials,
            clear: &mut clear,
            orbit: &mut orbit,
            errors: Vec::new(),
        };

        execute_root(&state.root, &mut ctx);

        for err in ctx.errors {
            error_writer.write(err);
        }

        // Stash the state for UI inspection (replaces any previous state).
        state_res.0 = Some(state);
    }
}

fn parse_state(event: &LoadMvsEvent) -> Result<State, MvsError> {
    match event {
        LoadMvsEvent::FromString(s) => {
            State::from_str(s).map_err(|e| MvsError::ParseError(e.to_string()))
        }
        LoadMvsEvent::FromFile(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| MvsError::ParseError(format!("{}: {e}", path.display())))?;
            State::from_str(&text).map_err(|e| MvsError::ParseError(e.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Tree traversal
// ---------------------------------------------------------------------------

fn children(node: &Node) -> &[Node] {
    node.children.as_deref().unwrap_or(&[])
}

fn execute_root(root: &Node, ctx: &mut Ctx) {
    // Canvas and camera are applied first so that any `focus` node encountered later
    // (deep inside a structure) overrides the camera — focus always wins.
    for child in children(root) {
        if child.kind == KindT::Canvas {
            if let Some(NodeParams::CanvasParams(p)) = &child.params {
                apply_canvas(p, ctx);
            }
        }
    }
    for child in children(root) {
        if child.kind == KindT::Camera {
            if let Some(NodeParams::CameraParams(p)) = &child.params {
                apply_camera(p, ctx);
            }
        }
    }

    for child in children(root) {
        match child.kind {
            KindT::GenericVisuals => execute_generic_visuals(child, ctx),
            KindT::Download => execute_download(child, ctx),
            _ => {}
        }
    }
}

fn execute_download(node: &Node, ctx: &mut Ctx) {
    let url = match &node.params {
        Some(NodeParams::DownloadParams(p)) => p.url.clone(),
        _ => return,
    };

    if is_http_url(&url) {
        ctx.errors.push(MvsError::UnsupportedNode {
            kind: KindT::Download,
            reason: format!("HTTP(S) downloads are not supported in this MVP: {url}"),
        });
        return;
    }

    let path = PathBuf::from(&url);
    let model = match load_model(&path) {
        Ok(m) => m,
        Err(_) => {
            ctx.errors.push(MvsError::FileNotFound(path));
            return;
        }
    };
    let ac = AtomCollection::from(&model);

    // download -> parse -> structure
    for parse in children(node) {
        if parse.kind != KindT::Parse {
            continue;
        }
        // `format` is informational here; load_model auto-detects from the file.
        if let Some(NodeParams::ParseParams(pp)) = &parse.params {
            if matches!(pp.format, ParseFormatT::Bcif) {
                ctx.errors.push(MvsError::UnsupportedNode {
                    kind: KindT::Parse,
                    reason: "bcif parsing is not supported".to_string(),
                });
            }
        }
        for structure in children(parse) {
            if structure.kind == KindT::Structure {
                execute_structure(structure, &model, &ac, ctx);
            }
        }
    }
}

fn execute_structure(node: &Node, model: &Model, ac: &AtomCollection, ctx: &mut Ctx) {
    // Assembly / symmetry are not expanded; warn and render the plain model.
    if let Some(NodeParams::StructureParams(sp)) = &node.params {
        report_unsupported_structure(sp, ctx);
    }

    // A `transform` node is a sibling of the components and applies to the whole
    // structure. Collect it first so it can be handed to every component.
    let mut pending_transform = Transform::IDENTITY;
    for child in children(node) {
        if child.kind == KindT::Transform {
            if let Some(NodeParams::TransformParams(tp)) = &child.params {
                match parse_transform(tp) {
                    Ok(t) => pending_transform = t,
                    Err(()) => ctx.errors.push(MvsError::InvalidTransform),
                }
            }
        }
    }

    for child in children(node) {
        match child.kind {
            KindT::Component => execute_component(child, model, ac, pending_transform, ctx),
            KindT::ComponentFromUri | KindT::ComponentFromSource => {
                ctx.errors.push(MvsError::UnsupportedNode {
                    kind: child.kind.clone(),
                    reason: "annotation-driven components are not supported".to_string(),
                });
            }
            KindT::LabelFromUri
            | KindT::LabelFromSource
            | KindT::TooltipFromUri
            | KindT::TooltipFromSource => {
                ctx.errors.push(MvsError::UnsupportedNode {
                    kind: child.kind.clone(),
                    reason: "annotation-driven labels/tooltips are not supported".to_string(),
                });
            }
            _ => {}
        }
    }
}

fn execute_component(
    node: &Node,
    model: &Model,
    ac: &AtomCollection,
    transform: Transform,
    ctx: &mut Ctx,
) {
    let selector = match &node.params {
        Some(NodeParams::ComponentInlineParams(ComponentInlineParams { selector })) => selector,
        _ => return,
    };

    let mask = evaluate_selector(selector, ac);
    if mask.all_false() {
        ctx.errors.push(MvsError::NoAtomsSelected {
            selector_description: describe_selector(selector),
        });
        return;
    }

    // A component may carry a representation, a label, and/or a focus child.
    for child in children(node) {
        match child.kind {
            KindT::Representation => render_representation(child, model, ac, &mask, transform, ctx),
            KindT::Label => {
                if let Some(NodeParams::LabelInlineParams(LabelInlineParams { text })) =
                    &child.params
                {
                    let centroid = centroid_of_masked(ac, &mask).unwrap_or(Vec3::ZERO);
                    ctx.commands.spawn((
                        MvsLabel(text.clone()),
                        Transform::from_translation(centroid),
                        MvsEntity,
                    ));
                }
            }
            KindT::Focus => {
                let params = match &child.params {
                    Some(NodeParams::FocusInlineParams(p)) => p.clone(),
                    _ => FocusInlineParams {
                        direction: None,
                        up: None,
                    },
                };
                apply_focus(&params, ac, &mask, ctx);
            }
            _ => {}
        }
    }
}

fn render_representation(
    repr: &Node,
    model: &Model,
    ac: &AtomCollection,
    mask: &AtomMask,
    transform: Transform,
    ctx: &mut Ctx,
) {
    let repr_type = match &repr.params {
        Some(NodeParams::RepresentationParams(RepresentationParams {
            representation_type,
        })) => representation_type.clone(),
        _ => return,
    };

    let (render_opt, degraded) = map_representation(&repr_type);
    if degraded {
        warn!("Surface representation is not supported; rendering as VdW spheres");
    }

    // Collect layered color nodes (applied in document order; later wins).
    let mut color_nodes: Vec<(ComponentSelector, ColorT)> = Vec::new();
    for child in children(repr) {
        match child.kind {
            KindT::Color => {
                if let Some(NodeParams::ColorInlineParams(cp)) = &child.params {
                    color_nodes.push((cp.base.selector.clone(), cp.color.clone()));
                }
            }
            KindT::ColorFromUri | KindT::ColorFromSource => {
                ctx.errors.push(MvsError::UnsupportedNode {
                    kind: child.kind.clone(),
                    reason: "annotation-driven coloring is not supported".to_string(),
                });
            }
            _ => {}
        }
    }

    let structure = Structure::builder()
        .pdb(model.clone())
        .rendertype(render_opt.clone())
        .build();

    let (mut mesh, vertex_map, map_kind) = match render_opt {
        RenderOptions::Cartoon | RenderOptions::Putty => {
            let (m, v) = structure.to_mesh_with_residue_map(Some(mask));
            (m, v, VertexMapKind::ResidueMap)
        }
        _ => {
            let (m, v) = structure.to_mesh_with_atom_map(Some(mask));
            (m, v, VertexMapKind::AtomMap)
        }
    };

    apply_mvs_colors(&mut mesh, &color_nodes, ac, &vertex_map, map_kind);

    ctx.commands.spawn((
        Mesh3d(ctx.meshes.add(mesh)),
        MeshMaterial3d(ctx.materials.add(StandardMaterial::default())),
        transform,
        MvsEntity,
    ));
}

fn execute_generic_visuals(node: &Node, ctx: &mut Ctx) {
    for child in children(node) {
        match child.kind {
            KindT::Sphere => {
                if let Some(NodeParams::SphereParams(p)) = &child.params {
                    spawn_sphere(p, ctx);
                }
            }
            KindT::Line => {
                if let Some(NodeParams::LineParams(p)) = &child.params {
                    spawn_line(p, ctx);
                }
            }
            _ => {}
        }
    }
}

fn spawn_sphere(p: &SphereParams, ctx: &mut Ctx) {
    let center = tuple_to_vec3(p.position);
    let mesh = Sphere::new(p.radius as f32).mesh().build();
    let material = StandardMaterial {
        base_color: color_t_to_bevy(&p.color),
        ..default()
    };
    ctx.commands.spawn((
        Mesh3d(ctx.meshes.add(mesh)),
        MeshMaterial3d(ctx.materials.add(material)),
        Transform::from_translation(center),
        MvsEntity,
    ));
}

fn spawn_line(p: &LineParams, ctx: &mut Ctx) {
    let p1 = tuple_to_vec3(p.position1);
    let p2 = tuple_to_vec3(p.position2);
    let center = (p1 + p2) * 0.5;
    let direction = p2 - p1;
    let height = direction.length().max(1e-6);
    let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize_or_zero());
    let mesh = Cylinder {
        radius: p.radius as f32,
        half_height: height * 0.5,
    }
    .mesh()
    .build();
    let material = StandardMaterial {
        base_color: color_t_to_bevy(&p.color),
        ..default()
    };
    ctx.commands.spawn((
        Mesh3d(ctx.meshes.add(mesh)),
        MeshMaterial3d(ctx.materials.add(material)),
        Transform {
            translation: center,
            rotation,
            ..default()
        },
        MvsEntity,
    ));
}

// ---------------------------------------------------------------------------
// Camera / canvas / focus application
// ---------------------------------------------------------------------------

fn apply_canvas(p: &CanvasParams, ctx: &mut Ctx) {
    *ctx.clear = ClearColor(color_t_to_bevy(&p.background_color));
}

fn apply_camera(p: &CameraParams, ctx: &mut Ctx) {
    let target = tuple_to_vec3(p.target);
    let position = tuple_to_vec3(p.position);
    set_orbit_from_position(ctx.orbit, target, position);
    if let Some(up) = p.up {
        ctx.orbit.up = tuple_to_vec3(up);
    }
}

fn apply_focus(params: &FocusInlineParams, ac: &AtomCollection, mask: &AtomMask, ctx: &mut Ctx) {
    let Some((center, diagonal)) = aabb_of_masked(ac, mask) else {
        return;
    };
    let radius = (diagonal * 1.5).max(1.0);
    ctx.orbit.focus = center;
    ctx.orbit.radius = radius;

    // An explicit view direction overrides the orbit orientation. `direction` is the
    // direction the camera looks *along*, so the camera sits opposite to it.
    if let Some(dir) = params.direction {
        let view = tuple_to_vec3(dir);
        if view.length() > 1e-6 {
            let offset = -view.normalize() * radius;
            set_yaw_pitch_from_offset(ctx.orbit, offset);
        }
    }
    if let Some(up) = params.up {
        ctx.orbit.up = tuple_to_vec3(up);
    }
}

fn set_orbit_from_position(orbit: &mut OrbitCamera, target: Vec3, position: Vec3) {
    orbit.focus = target;
    let offset = position - target;
    let r = offset.length();
    if r > 1e-6 {
        orbit.radius = r;
        set_yaw_pitch_from_offset(orbit, offset);
    }
}

fn set_yaw_pitch_from_offset(orbit: &mut OrbitCamera, offset: Vec3) {
    let r = offset.length();
    if r > 1e-6 {
        orbit.yaw = offset.x.atan2(offset.z);
        orbit.pitch = (offset.y / r).clamp(-1.0, 1.0).asin();
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (unit tested)
// ---------------------------------------------------------------------------

/// `true` if `url` is an HTTP(S) URL (deferred in the MVP); otherwise treat as a
/// local filesystem path.
fn is_http_url(url: &str) -> bool {
    let u = url.trim_start();
    u.starts_with("http://") || u.starts_with("https://")
}

/// Map a MVS representation type to a renderer. The bool is `true` when the type was
/// degraded (Surface → Solid) and a warning should be surfaced.
fn map_representation(repr: &RepresentationTypeT) -> (RenderOptions, bool) {
    match repr {
        RepresentationTypeT::Cartoon => (RenderOptions::Cartoon, false),
        RepresentationTypeT::BallAndStick => (RenderOptions::BallAndStick, false),
        RepresentationTypeT::Surface => (RenderOptions::Solid, true),
    }
}

/// Parse a [`TransformParams`] into a Bevy [`Transform`]. The rotation, when present,
/// must be a 9-element column-major 3×3 matrix; any other length is an error.
fn parse_transform(params: &TransformParams) -> Result<Transform, ()> {
    let mut transform = Transform::IDENTITY;
    if let Some(rot) = &params.rotation {
        if rot.len() != 9 {
            return Err(());
        }
        let m = Mat3::from_cols_array(&[
            rot[0] as f32,
            rot[1] as f32,
            rot[2] as f32,
            rot[3] as f32,
            rot[4] as f32,
            rot[5] as f32,
            rot[6] as f32,
            rot[7] as f32,
            rot[8] as f32,
        ]);
        transform.rotation = Quat::from_mat3(&m);
    }
    if let Some(t) = &params.translation {
        transform.translation = tuple_to_vec3(*t);
    }
    Ok(transform)
}

/// Axis-aligned bounding box of the masked atoms: returns `(center, diagonal_length)`.
/// `None` when no atom passes the mask.
fn aabb_of_masked(ac: &AtomCollection, mask: &AtomMask) -> Option<(Vec3, f32)> {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;
    for (i, coord) in ac.get_coords().iter().enumerate() {
        if !mask.0.get(i).copied().unwrap_or(false) {
            continue;
        }
        any = true;
        let p = Vec3::from_array(*coord);
        min = min.min(p);
        max = max.max(p);
    }
    if !any {
        return None;
    }
    Some(((min + max) * 0.5, (max - min).length()))
}

/// Mean position of the masked atoms; `None` when no atom passes the mask.
fn centroid_of_masked(ac: &AtomCollection, mask: &AtomMask) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for (i, coord) in ac.get_coords().iter().enumerate() {
        if !mask.0.get(i).copied().unwrap_or(false) {
            continue;
        }
        sum += Vec3::from_array(*coord);
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

fn report_unsupported_structure(sp: &StructureParams, ctx: &mut Ctx) {
    let reason = match sp.structure_type {
        StructureTypeT::Assembly => Some("assembly biological units are not expanded"),
        StructureTypeT::Symmetry | StructureTypeT::SymmetryMates => {
            Some("symmetry mates are not generated")
        }
        StructureTypeT::Model => None,
    };
    if let Some(reason) = reason {
        ctx.errors.push(MvsError::UnsupportedNode {
            kind: KindT::Structure,
            reason: format!("{reason}; rendering the asymmetric unit"),
        });
    }
}

fn describe_selector(selector: &ComponentSelector) -> String {
    match selector {
        ComponentSelector::Selector(s) => format!("{s:?}"),
        ComponentSelector::Expression(e) => format!("{e:?}"),
        ComponentSelector::ExpressionList(l) => format!("expression list ({} items)", l.len()),
    }
}

fn tuple_to_vec3(t: (f64, f64, f64)) -> Vec3 {
    Vec3::new(t.0 as f32, t.1 as f32, t.2 as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::Messages;
    use ferritin_core::info::elements::Element;
    use ferritin_molviewspec::molviewspec::nodes::ComponentSelectorT;

    // ---------------------------------------------------------------------------
    // Headless integration test helpers
    // ---------------------------------------------------------------------------

    fn make_app() -> App {
        let mut app = App::new();
        app.add_plugins(bevy::MinimalPlugins);
        app.add_plugins(bevy::asset::AssetPlugin::default());
        app.add_plugins(MvsPlugin);
        app
    }

    fn count_meshes(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<Entity, (With<Mesh3d>, With<MvsEntity>)>()
            .iter(app.world())
            .count()
    }

    fn count_labels(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<Entity, With<MvsLabel>>()
            .iter(app.world())
            .count()
    }

    fn get_errors(app: &App) -> Vec<MvsError> {
        app.world()
            .resource::<Messages<MvsError>>()
            .iter_current_update_messages()
            .cloned()
            .collect()
    }

    fn send_and_update(app: &mut App, json: &str) {
        app.world_mut()
            .write_message(LoadMvsEvent::FromString(json.to_string()));
        app.update();
    }

    /// Minimal MVSJ with one structure component using `ball_and_stick` (no residue map needed).
    fn mvsj_one_component(path: &str, selector: &str, repr: &str) -> String {
        format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "model"}},
                      "children": [{{
                        "kind": "component",
                        "params": {{"selector": "{selector}"}},
                        "children": [{{
                          "kind": "representation",
                          "params": {{"type": "{repr}"}}
                        }}]
                      }}]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        )
    }

    // --- Pure-helper unit tests (T3-01 .. T3-09) ---------------------------

    #[test]
    fn test_url_is_http_https() {
        assert!(is_http_url("https://files.wwpdb.org/download/1cbs.cif"));
        assert!(is_http_url("http://example.com/foo.cif"));
    }

    #[test]
    fn test_url_is_local() {
        assert!(!is_http_url("data/structures/1cbs.cif"));
        assert!(!is_http_url("/absolute/path/1cbs.cif"));
        assert!(!is_http_url("./relative/1cbs.cif"));
    }

    /// 3 atoms at (0,0,0), (2,0,0), (1,3,0); all selected.
    fn three_atom_collection() -> AtomCollection {
        let coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]];
        let res_ids = vec![1, 1, 1];
        let res_names = vec!["ALA".to_string(); 3];
        let is_hetero = vec![false; 3];
        let elements = vec![Element::N, Element::C, Element::C];
        let atom_names = vec!["N".to_string(), "CA".to_string(), "C".to_string()];
        let chain_ids = vec!["A".to_string(); 3];
        AtomCollection::new(
            3, coords, res_ids, res_names, is_hetero, elements, atom_names, chain_ids, None,
        )
    }

    #[test]
    fn test_aabb_from_masked_atoms() {
        let ac = three_atom_collection();
        let mask = AtomMask(vec![true; 3]);
        let (center, diag) = aabb_of_masked(&ac, &mask).unwrap();
        assert!((center - Vec3::new(1.0, 1.5, 0.0)).length() < 1e-5);
        assert!((diag - 13.0_f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_aabb_empty_mask_is_none() {
        let ac = three_atom_collection();
        let mask = AtomMask(vec![false; 3]);
        assert!(aabb_of_masked(&ac, &mask).is_none());
    }

    #[test]
    fn test_centroid_from_masked_atoms() {
        let ac = three_atom_collection();
        let mask = AtomMask(vec![true; 3]);
        let c = centroid_of_masked(&ac, &mask).unwrap();
        assert!((c - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn test_centroid_single_atom() {
        let coords = vec![[5.0, 5.0, 5.0]];
        let ac = AtomCollection::new(
            1,
            coords,
            vec![1],
            vec!["ALA".to_string()],
            vec![false],
            vec![Element::C],
            vec!["CA".to_string()],
            vec!["A".to_string()],
            None,
        );
        let mask = AtomMask(vec![true]);
        let c = centroid_of_masked(&ac, &mask).unwrap();
        assert_eq!(c, Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_map_repr_cartoon() {
        let (opt, degraded) = map_representation(&RepresentationTypeT::Cartoon);
        assert!(matches!(opt, RenderOptions::Cartoon));
        assert!(!degraded);
    }

    #[test]
    fn test_map_repr_ball_and_stick() {
        let (opt, degraded) = map_representation(&RepresentationTypeT::BallAndStick);
        assert!(matches!(opt, RenderOptions::BallAndStick));
        assert!(!degraded);
    }

    #[test]
    fn test_map_repr_surface_degrades() {
        let (opt, degraded) = map_representation(&RepresentationTypeT::Surface);
        assert!(matches!(opt, RenderOptions::Solid));
        assert!(degraded, "Surface must signal degradation");
    }

    #[test]
    fn test_rotation_matrix_wrong_length() {
        let params = TransformParams {
            rotation: Some(vec![1.0; 8]),
            translation: None,
        };
        assert_eq!(parse_transform(&params), Err(()));
    }

    #[test]
    fn test_transform_identity_rotation_and_translation() {
        let params = TransformParams {
            rotation: Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            translation: Some((1.0, 2.0, 3.0)),
        };
        let t = parse_transform(&params).unwrap();
        assert!((t.translation - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        assert!(t.rotation.angle_between(Quat::IDENTITY) < 1e-4);
    }

    #[test]
    fn test_selector_description_does_not_panic() {
        let s = describe_selector(&ComponentSelector::Selector(ComponentSelectorT::Water));
        assert!(s.contains("Water"));
    }

    // --- Headless Bevy integration tests (T3-10 .. T3-31) ------------------

    // T3-10: A minimal valid state spawns exactly one mesh entity.
    #[test]
    fn test_basic_load_spawns_one_mesh() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(
            &mut app,
            &mvsj_one_component(&path, "all", "ball_and_stick"),
        );
        assert_eq!(count_meshes(&mut app), 1, "T3-10: expected 1 mesh entity");
    }

    // T3-11: Two components (protein + water) with a focus node → 2 meshes.
    #[test]
    fn test_two_components_two_meshes() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "model"}},
                      "children": [
                        {{
                          "kind": "component",
                          "params": {{"selector": "protein"}},
                          "children": [
                            {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}},
                            {{"kind": "focus", "params": {{}}}}
                          ]
                        }},
                        {{
                          "kind": "component",
                          "params": {{"selector": "water"}},
                          "children": [
                            {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}
                          ]
                        }}
                      ]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        assert_eq!(
            count_meshes(&mut app),
            2,
            "T3-11: protein + water → 2 meshes"
        );
    }

    // T3-13: A label node spawns an MvsLabel entity.
    #[test]
    fn test_label_spawns_entity() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "model"}},
                      "children": [{{
                        "kind": "component",
                        "params": {{"selector": "all"}},
                        "children": [
                          {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}},
                          {{"kind": "label", "params": {{"text": "Hello World"}}}}
                        ]
                      }}]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        assert_eq!(
            count_labels(&mut app),
            1,
            "T3-13: expected 1 MvsLabel entity"
        );
    }

    // T3-14: A focus node repositions OrbitCamera away from its default.
    #[test]
    fn test_focus_repositions_camera() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "model"}},
                      "children": [{{
                        "kind": "component",
                        "params": {{"selector": "all"}},
                        "children": [
                          {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}},
                          {{"kind": "focus", "params": {{}}}}
                        ]
                      }}]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        let default_orbit = OrbitCamera::default();
        send_and_update(&mut app, &json);
        let orbit = app.world().resource::<OrbitCamera>().clone();
        // 101m.cif atoms are not all at origin; focus must shift the radius.
        assert_ne!(
            orbit.radius, default_orbit.radius,
            "T3-14: focus should change orbit radius"
        );
    }

    // T3-15: A canvas node sets ClearColor.
    #[test]
    fn test_canvas_sets_clear_color() {
        let json = r#"{
          "root": {
            "kind": "root",
            "children": [{
              "kind": "canvas",
              "params": {"background_color": "red"}
            }]
          },
          "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00"}
        }"#;
        let mut app = make_app();
        send_and_update(&mut app, json);
        let cc = app.world().resource::<ClearColor>();
        // red → (1,0,0,1)
        let r = cc.0.to_linear();
        assert!(
            r.red > 0.9 && r.green < 0.1,
            "T3-15: canvas must set red background"
        );
    }

    // T3-16: A camera node sets OrbitCamera focus/radius.
    #[test]
    fn test_camera_node_sets_orbit() {
        let json = r#"{
          "root": {
            "kind": "root",
            "children": [{
              "kind": "camera",
              "params": {
                "target": [10.0, 20.0, 30.0],
                "position": [10.0, 20.0, 80.0],
                "up": [0.0, 1.0, 0.0]
              }
            }]
          },
          "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00"}
        }"#;
        let mut app = make_app();
        send_and_update(&mut app, json);
        let orbit = app.world().resource::<OrbitCamera>().clone();
        assert!(
            (orbit.focus - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-3,
            "T3-16: camera target must set orbit focus"
        );
        assert!(
            (orbit.radius - 50.0).abs() < 1e-3,
            "T3-16: distance from target to position = 50"
        );
    }

    // T3-17: HTTP URL emits UnsupportedNode, no meshes spawned.
    #[test]
    fn test_http_url_emits_unsupported_error() {
        let json = r#"{
          "root": {
            "kind": "root",
            "children": [{
              "kind": "download",
              "params": {"url": "https://files.wwpdb.org/download/1cbs.cif"},
              "children": [{"kind": "parse", "params": {"format": "mmcif"}, "children": []}]
            }]
          },
          "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00"}
        }"#;
        let mut app = make_app();
        send_and_update(&mut app, json);
        let errors = get_errors(&app);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                MvsError::UnsupportedNode {
                    kind: KindT::Download,
                    ..
                }
            )),
            "T3-17: HTTP download must emit UnsupportedNode"
        );
        assert_eq!(count_meshes(&mut app), 0, "T3-17: no meshes for http");
    }

    // T3-18: Bad file path emits FileNotFound, no meshes spawned.
    #[test]
    fn test_bad_path_emits_file_not_found() {
        let json = r#"{
          "root": {
            "kind": "root",
            "children": [{
              "kind": "download",
              "params": {"url": "/nonexistent/path/nope.cif"},
              "children": [{"kind": "parse", "params": {"format": "mmcif"}, "children": []}]
            }]
          },
          "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00"}
        }"#;
        let mut app = make_app();
        send_and_update(&mut app, json);
        let errors = get_errors(&app);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, MvsError::FileNotFound(_))),
            "T3-18: bad path must emit FileNotFound"
        );
        assert_eq!(
            count_meshes(&mut app),
            0,
            "T3-18: no meshes on file not found"
        );
    }

    // T3-19: Empty selector emits NoAtomsSelected but sibling component continues.
    #[test]
    fn test_empty_selector_emits_error_and_sibling_continues() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        // "nucleic" selector on 101m.cif (a myoglobin) should match zero atoms.
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "model"}},
                      "children": [
                        {{
                          "kind": "component",
                          "params": {{"selector": "nucleic"}},
                          "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                        }},
                        {{
                          "kind": "component",
                          "params": {{"selector": "all"}},
                          "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                        }}
                      ]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        let errors = get_errors(&app);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, MvsError::NoAtomsSelected { .. })),
            "T3-19: empty selector must emit NoAtomsSelected"
        );
        // The sibling "all" component must still render.
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-19: sibling component must still render"
        );
    }

    // T3-20: Assembly structure type emits UnsupportedNode but still renders the asymmetric unit.
    #[test]
    fn test_assembly_emits_warning_and_renders() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "assembly"}},
                      "children": [{{
                        "kind": "component",
                        "params": {{"selector": "all"}},
                        "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                      }}]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        let errors = get_errors(&app);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                MvsError::UnsupportedNode {
                    kind: KindT::Structure,
                    ..
                }
            )),
            "T3-20: assembly must emit UnsupportedNode"
        );
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-20: assembly still renders asymmetric unit"
        );
    }

    // T3-21: Symmetry structure type emits UnsupportedNode but renders.
    #[test]
    fn test_symmetry_emits_warning_and_renders() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "symmetry"}},
                      "children": [{{
                        "kind": "component",
                        "params": {{"selector": "all"}},
                        "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                      }}]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        let errors = get_errors(&app);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                MvsError::UnsupportedNode {
                    kind: KindT::Structure,
                    ..
                }
            )),
            "T3-21: symmetry must emit UnsupportedNode"
        );
        assert_eq!(count_meshes(&mut app), 1, "T3-21: symmetry still renders");
    }

    // T3-22: component_from_uri emits UnsupportedNode and execution continues.
    #[test]
    fn test_component_from_uri_emits_unsupported() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "model"}},
                      "children": [
                        {{
                          "kind": "component_from_uri",
                          "params": {{"uri": "https://example.com/sel.json", "format": "json", "schema": "all_atomic"}},
                          "children": []
                        }},
                        {{
                          "kind": "component",
                          "params": {{"selector": "all"}},
                          "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                        }}
                      ]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        let errors = get_errors(&app);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                MvsError::UnsupportedNode {
                    kind: KindT::ComponentFromUri,
                    ..
                }
            )),
            "T3-22: component_from_uri must emit UnsupportedNode"
        );
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-22: sibling component still renders"
        );
    }

    // T3-23: Scene clear on reload — second load replaces first.
    #[test]
    fn test_scene_clear_on_reload() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = mvsj_one_component(&path, "all", "ball_and_stick");
        let mut app = make_app();

        // First load.
        send_and_update(&mut app, &json);
        assert_eq!(count_meshes(&mut app), 1, "T3-23: first load spawns 1 mesh");

        // Second load with a different (water) selector.
        let json2 = mvsj_one_component(&path, "water", "ball_and_stick");
        send_and_update(&mut app, &json2);
        // Should still be 1 (old entity despawned, new one spawned).
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-23: second load replaces first, still 1 mesh"
        );
    }

    // T3-24: A generic_visuals sphere spawns exactly one MvsEntity.
    #[test]
    fn test_generic_sphere_spawns_entity() {
        let json = r#"{
          "root": {
            "kind": "root",
            "children": [{
              "kind": "generic_visuals",
              "children": [{
                "kind": "sphere",
                "params": {
                  "position": [0.0, 0.0, 0.0],
                  "radius": 1.0,
                  "color": "red",
                  "label": "test"
                }
              }]
            }]
          },
          "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00"}
        }"#;
        let mut app = make_app();
        send_and_update(&mut app, json);
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-24: sphere must spawn 1 MvsEntity mesh"
        );
    }

    // T3-25: A generic_visuals line spawns exactly one MvsEntity.
    #[test]
    fn test_generic_line_spawns_entity() {
        let json = r#"{
          "root": {
            "kind": "root",
            "children": [{
              "kind": "generic_visuals",
              "children": [{
                "kind": "line",
                "params": {
                  "position1": [0.0, 0.0, 0.0],
                  "position2": [1.0, 1.0, 1.0],
                  "radius": 0.1,
                  "color": "blue",
                  "label": "line"
                }
              }]
            }]
          },
          "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00"}
        }"#;
        let mut app = make_app();
        send_and_update(&mut app, json);
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-25: line must spawn 1 MvsEntity mesh"
        );
    }

    // T3-26: A transform node applies to all components (transform is non-identity).
    #[test]
    fn test_transform_applied_to_component() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [{{
                  "kind": "download",
                  "params": {{"url": "{path}"}},
                  "children": [{{
                    "kind": "parse",
                    "params": {{"format": "mmcif"}},
                    "children": [{{
                      "kind": "structure",
                      "params": {{"type": "model"}},
                      "children": [
                        {{
                          "kind": "transform",
                          "params": {{
                            "rotation": [1,0,0, 0,1,0, 0,0,1],
                            "translation": [10.0, 0.0, 0.0]
                          }}
                        }},
                        {{
                          "kind": "component",
                          "params": {{"selector": "all"}},
                          "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                        }}
                      ]
                    }}]
                  }}]
                }}]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        // The mesh should be spawned (transform is valid identity-ish rotation + translation).
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-26: transform + component → 1 mesh"
        );
        // Verify no InvalidTransform error.
        let errors = get_errors(&app);
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, MvsError::InvalidTransform)),
            "T3-26: valid transform must not emit InvalidTransform"
        );
    }

    // T3-27: No transform node → identity transform (no error, mesh spawned).
    #[test]
    fn test_no_transform_uses_identity() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(
            &mut app,
            &mvsj_one_component(&path, "all", "ball_and_stick"),
        );
        let errors = get_errors(&app);
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, MvsError::InvalidTransform)),
            "T3-27: no transform node must not emit InvalidTransform"
        );
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-27: no transform → mesh spawned"
        );
    }

    // T3-28: Two download nodes → 2 meshes total.
    #[test]
    fn test_two_downloads_two_meshes() {
        use ferritin_test_data::TestFile;
        // Both must be CIF files; load_model does not support PDB.
        let (path1, _tmp1) = TestFile::protein_01().create_temp().unwrap();
        let (path2, _tmp2) = TestFile::protein_03().create_temp().unwrap();
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [
                  {{
                    "kind": "download",
                    "params": {{"url": "{path1}"}},
                    "children": [{{
                      "kind": "parse",
                      "params": {{"format": "mmcif"}},
                      "children": [{{
                        "kind": "structure",
                        "params": {{"type": "model"}},
                        "children": [{{
                          "kind": "component",
                          "params": {{"selector": "all"}},
                          "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                        }}]
                      }}]
                    }}]
                  }},
                  {{
                    "kind": "download",
                    "params": {{"url": "{path2}"}},
                    "children": [{{
                      "kind": "parse",
                      "params": {{"format": "mmcif"}},
                      "children": [{{
                        "kind": "structure",
                        "params": {{"type": "model"}},
                        "children": [{{
                          "kind": "component",
                          "params": {{"selector": "all"}},
                          "children": [{{"kind": "representation", "params": {{"type": "ball_and_stick"}}}}]
                        }}]
                      }}]
                    }}]
                  }}
                ]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        assert_eq!(count_meshes(&mut app), 2, "T3-28: two downloads → 2 meshes");
    }

    // T3-29: MvsStateResource is populated after a successful load.
    #[test]
    fn test_mvs_state_resource_set() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(
            &mut app,
            &mvsj_one_component(&path, "all", "ball_and_stick"),
        );
        let state_res = app.world().resource::<MvsStateResource>();
        assert!(
            state_res.0.is_some(),
            "T3-29: MvsStateResource must be Some after successful load"
        );
    }

    // T3-30: MvsStateResource is replaced on second load.
    #[test]
    fn test_mvs_state_resource_replaced() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(
            &mut app,
            &mvsj_one_component(&path, "all", "ball_and_stick"),
        );
        send_and_update(
            &mut app,
            &mvsj_one_component(&path, "water", "ball_and_stick"),
        );
        let state_res = app.world().resource::<MvsStateResource>();
        assert!(
            state_res.0.is_some(),
            "T3-30: MvsStateResource must remain Some after second load"
        );
    }

    // T3-31: Camera node followed by a focus node → focus wins (radius differs from camera).
    #[test]
    fn test_camera_then_focus_focus_wins() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        // Camera sets radius=1000 (far target); focus will override with a tighter AABB radius.
        let json = format!(
            r#"{{
              "root": {{
                "kind": "root",
                "children": [
                  {{
                    "kind": "camera",
                    "params": {{
                      "target": [0.0, 0.0, 0.0],
                      "position": [0.0, 0.0, 1000.0],
                      "up": [0.0, 1.0, 0.0]
                    }}
                  }},
                  {{
                    "kind": "download",
                    "params": {{"url": "{path}"}},
                    "children": [{{
                      "kind": "parse",
                      "params": {{"format": "mmcif"}},
                      "children": [{{
                        "kind": "structure",
                        "params": {{"type": "model"}},
                        "children": [{{
                          "kind": "component",
                          "params": {{"selector": "all"}},
                          "children": [
                            {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}},
                            {{"kind": "focus", "params": {{}}}}
                          ]
                        }}]
                      }}]
                    }}]
                  }}
                ]
              }},
              "metadata": {{"version": "1.0", "timestamp": "2024-01-01T00:00:00"}}
            }}"#
        );
        let mut app = make_app();
        send_and_update(&mut app, &json);
        let orbit = app.world().resource::<OrbitCamera>().clone();
        // Focus-computed radius from AABB of 101m.cif will be << 1000.
        assert!(
            orbit.radius < 500.0,
            "T3-31: focus must override camera; radius={} should be << 1000",
            orbit.radius
        );
    }
}
