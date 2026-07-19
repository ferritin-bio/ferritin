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
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel};
use bevy::prelude::*;
use bevy::ui::Node as UiNode;
use ferritin_core::{load_model, AtomCollection, Model};
use ferritin_molviewspec::molviewspec::nodes::{
    CameraParams, CanvasParams, ColorT, ColorThemeT, ComponentInlineParams, ComponentSelector,
    FocusInlineParams, KindT, LabelInlineParams, LineParams, Node, NodeParams, ParseFormatT,
    RepresentationParams, RepresentationTypeT, SphereParams, State, StructureParams,
    StructureTypeT, TransformParams,
};

use crate::colors::{apply_mvs_colors_with_theme, color_t_to_bevy, VertexMapKind};
use crate::selection::{evaluate_selector, AtomMask};
use crate::structure::{RenderOptions, Structure};

// ---------------------------------------------------------------------------
// Public ECS types
// ---------------------------------------------------------------------------

/// Marker attached to every entity spawned by the executor. On each reload old
/// `MvsEntity` entities are hidden, marked retired, then despawned shortly after
/// the replacement scene is built.
#[derive(Component)]
pub struct MvsEntity;

/// Old scene entities are hidden first and despawned a frame later. This keeps
/// mesh/material handles alive long enough for Bevy's render allocator to observe
/// the scene transition without a same-frame free/reallocate churn.
#[derive(Component)]
struct MvsRetiredEntity {
    remaining_frames: u8,
}

/// A MVS `label` node, spawned at the centroid of its component's atoms.
/// [`update_mvs_label_billboards`] projects this world-space anchor to screen
/// space each frame and keeps a companion UI [`Text`] entity positioned there.
#[derive(Component)]
pub struct MvsLabel(pub String);

/// Links a [`MvsLabel`] anchor entity to its billboard UI [`Text`] entity.
#[derive(Component)]
struct MvsLabelUiNode(Entity);

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
    /// A representation type was requested but rendered as a different, degraded
    /// fallback (e.g. Surface -> VdW spheres). Previously only a `warn!()` log
    /// call, invisible to anyone using the interactive viewer (ferritin-ala.9).
    RepresentationDegraded {
        requested: RepresentationTypeT,
        rendered_as: &'static str,
    },
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
            .add_systems(
                Update,
                (
                    execute_mvs_on_load,
                    update_mvs_label_billboards,
                    apply_molecular_camera_defaults,
                    despawn_retired_mvs_entities,
                )
                    .chain(),
            );
    }
}

/// Applies MVS viewer camera defaults. Tonemapping is disabled so explicit
/// molecular colors remain legible (ferritin-ala.8), and SSAO is enabled to add
/// contact shadows/depth cues for dense all-atom scenes (ferritin-c1k.4).
fn apply_molecular_camera_defaults(
    mut commands: Commands,
    cameras: Query<Entity, Added<Camera3d>>,
) {
    for entity in &cameras {
        commands.entity(entity).insert((
            Tonemapping::None,
            ScreenSpaceAmbientOcclusion {
                quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
                constant_object_thickness: 0.25,
            },
        ));
    }
}

fn despawn_retired_mvs_entities(
    mut commands: Commands,
    mut retired: Query<(Entity, &mut MvsRetiredEntity)>,
) {
    for (entity, mut retired) in &mut retired {
        if retired.remaining_frames == 0 {
            commands.entity(entity).despawn();
        } else {
            retired.remaining_frames -= 1;
        }
    }
}

/// Projects each [`MvsLabel`] anchor's world position through the active 3D
/// camera and keeps a companion screen-space [`Text`] node positioned there,
/// creating it on first sight. Labels that project behind the camera or
/// outside the viewport are hidden rather than despawned, since the anchor
/// entity (and thus this link) is recreated on every scene reload anyway.
fn update_mvs_label_billboards(
    mut commands: Commands,
    camera_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    label_q: Query<(Entity, &Transform, Option<&MvsLabelUiNode>), With<MvsLabel>>,
    mut node_q: Query<&mut UiNode>,
    text_q: Query<&MvsLabel>,
) {
    let Ok((camera, camera_transform)) = camera_q.single() else {
        return;
    };

    for (entity, transform, ui_node) in &label_q {
        let viewport_pos = camera
            .world_to_viewport(camera_transform, transform.translation)
            .ok();

        match (ui_node, viewport_pos) {
            (Some(MvsLabelUiNode(ui_entity)), Some(pos)) => {
                if let Ok(mut node) = node_q.get_mut(*ui_entity) {
                    node.display = Display::Flex;
                    node.left = Val::Px(pos.x);
                    node.top = Val::Px(pos.y);
                }
            }
            (Some(MvsLabelUiNode(ui_entity)), None) => {
                if let Ok(mut node) = node_q.get_mut(*ui_entity) {
                    node.display = Display::None;
                }
            }
            (None, Some(pos)) => {
                let Ok(label) = text_q.get(entity) else {
                    continue;
                };
                let ui_entity = commands
                    .spawn((
                        Text::new(label.0.clone()),
                        TextColor(Color::WHITE),
                        UiNode {
                            position_type: PositionType::Absolute,
                            left: Val::Px(pos.x),
                            top: Val::Px(pos.y),
                            ..default()
                        },
                        MvsEntity,
                    ))
                    .id();
                commands.entity(entity).insert(MvsLabelUiNode(ui_entity));
            }
            (None, None) => {}
        }
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
    /// Union of every `focus()` node's AABB seen so far this load, so a scene with
    /// multiple focused components (e.g. a superposition) frames all of them
    /// instead of just the last one processed — see ferritin-ala.5.
    focus_union: Option<(Vec3, Vec3)>,
    /// Union of every *rendered* component's world-space AABB seen so far this
    /// load (regardless of whether it was focused). Used to keep a small
    /// focus() target (e.g. two ions inside a large protein) from pulling the
    /// camera in so close that the surrounding, still-rendered geometry
    /// near-clips — see ferritin-t0h.5.
    scene_bounds: Option<(Vec3, Vec3)>,
    /// Live viewport width/height, read from the render camera at load time.
    /// `fit_radius_for_view` needs this to compute the true horizontal FOV —
    /// using the (narrower) vertical FOV for both axes on a widescreen viewport
    /// over-estimated the required distance and left whole-structure framing
    /// filling under half the frame — see ferritin-t0h.10.
    aspect_ratio: f32,
}

/// Reacts to [`LoadMvsEvent`]: parse, clear the previous scene, execute the new tree.
#[allow(clippy::too_many_arguments)]
fn execute_mvs_on_load(
    mut load_events: MessageReader<LoadMvsEvent>,
    mut error_writer: MessageWriter<MvsError>,
    existing: Query<Entity, With<MvsEntity>>,
    camera_q: Query<&Camera, With<Camera3d>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut clear: ResMut<ClearColor>,
    mut orbit: ResMut<OrbitCamera>,
    mut state_res: ResMut<MvsStateResource>,
) {
    // Default to 16:9 (every example in this repo opens at that ratio) when no
    // camera/viewport is available yet; recomputed fresh per load so a resized
    // window is picked up on the next scene load.
    let aspect_ratio = camera_q
        .single()
        .ok()
        .and_then(|c| c.logical_viewport_size())
        .filter(|size| size.y > 0.0)
        .map(|size| size.x / size.y)
        .unwrap_or(16.0 / 9.0);

    for event in load_events.read() {
        // Parse first; a parse failure aborts only this event.
        let state = match parse_state(event) {
            Ok(s) => s,
            Err(e) => {
                error_writer.write(e);
                continue;
            }
        };

        // Retire the previous scene. Immediate same-frame despawn+spawn can trip
        // Bevy's mesh slab allocator during heavy MVS scene switches; hiding and
        // removing a frame later keeps the old asset handles alive through the
        // transition while the new scene is already visible.
        for entity in existing.iter() {
            commands.entity(entity).insert((
                Visibility::Hidden,
                MvsRetiredEntity {
                    remaining_frames: 1,
                },
            ));
        }

        let mut ctx = Ctx {
            commands: &mut commands,
            meshes: &mut meshes,
            materials: &mut materials,
            clear: &mut clear,
            orbit: &mut orbit,
            errors: Vec::new(),
            focus_union: None,
            scene_bounds: None,
            aspect_ratio,
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
                apply_focus(&params, ac, &mask, transform, ctx);
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
    // Track this component's world-space AABB in the running scene bounds so a
    // later, smaller focus() target can be checked against everything actually
    // rendered, not just itself (ferritin-t0h.5).
    if let Some((local_min, local_max)) = aabb_of_masked(ac, mask) {
        let (world_min, world_max) = transform_aabb(local_min, local_max, transform);
        ctx.scene_bounds = Some(match ctx.scene_bounds {
            Some((prev_min, prev_max)) => (prev_min.min(world_min), prev_max.max(world_max)),
            None => (world_min, world_max),
        });
    }

    let (repr_type, color_theme) = match &repr.params {
        Some(NodeParams::RepresentationParams(RepresentationParams {
            representation_type,
            color_theme,
        })) => (representation_type.clone(), color_theme.clone()),
        _ => return,
    };

    let (render_opt, degraded) = map_representation(&repr_type);
    if degraded {
        warn!("Surface representation is not supported; rendering as VdW spheres");
        ctx.errors.push(MvsError::RepresentationDegraded {
            requested: repr_type.clone(),
            rendered_as: "VdW spheres (Spacefill)",
        });
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

    // Use the explicit color_theme when present; fall back to ElementSymbol (CPK)
    // so structures without any Color child nodes get sensible default coloring.
    let effective_theme = color_theme.unwrap_or(ColorThemeT::ElementSymbol);
    apply_mvs_colors_with_theme(
        &mut mesh,
        &effective_theme,
        &color_nodes,
        ac,
        &vertex_map,
        map_kind,
    );

    // Wireframe (MVS "line") geometry is now thin triangle cylinders with real
    // normals (ferritin-t0h.9), so it can take normal PBR shading like every
    // other representation instead of being forced unlit.
    let mut material = StandardMaterial::default();
    // Surface silently degrades to VdW spheres (Solid); tint it amber so it reads as
    // a fallback rather than looking pixel-identical to a real Spacefill request —
    // the `degraded` warning was previously log-only, invisible in the viewer itself.
    if degraded {
        material.base_color = Color::srgb(1.0, 0.78, 0.5);
        material.emissive = LinearRgba::rgb(0.15, 0.08, 0.0);
    }

    ctx.commands.spawn((
        Mesh3d(ctx.meshes.add(mesh)),
        MeshMaterial3d(ctx.materials.add(material)),
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

fn apply_focus(
    params: &FocusInlineParams,
    ac: &AtomCollection,
    mask: &AtomMask,
    transform: Transform,
    ctx: &mut Ctx,
) {
    let Some((local_min, local_max)) = aabb_of_masked(ac, mask) else {
        return;
    };
    // The component's own transform (e.g. a superposition's translate) isn't baked
    // into `ac`'s coordinates, so it must be applied here or focus() targets the
    // pre-transform location.
    let (world_min, world_max) = transform_aabb(local_min, local_max, transform);

    // Union with every other focus() target seen so far this load, so a scene with
    // multiple focused components (e.g. a superposition) frames all of them instead
    // of just the last one processed.
    let (union_min, union_max) = match ctx.focus_union {
        Some((prev_min, prev_max)) => (prev_min.min(world_min), prev_max.max(world_max)),
        None => (world_min, world_max),
    };
    ctx.focus_union = Some((union_min, union_max));

    ctx.orbit.focus = (union_min + union_max) * 0.5;

    if let Some(up) = params.up {
        ctx.orbit.up = tuple_to_vec3(up);
    }
    // An explicit view direction overrides the orbit orientation. `direction` is the
    // direction the camera looks *along*, so the camera sits opposite to it. Resolve
    // this before fitting the radius, since the fit depends on the final yaw/pitch.
    if let Some(dir) = params.direction {
        let view = tuple_to_vec3(dir);
        if view.length() > 1e-6 {
            set_yaw_pitch_from_offset(ctx.orbit, -view.normalize());
        }
    }

    let mut radius = fit_radius_for_view(union_min, union_max, ctx.orbit, ctx.aspect_ratio);

    // Clamp against the whole rendered scene, not just the focus target: a tiny
    // target (e.g. two ions) deep inside a much larger structure would otherwise
    // pull the camera in close enough that the surrounding, still-rendered
    // geometry near-clips (ferritin-t0h.5). Re-running the same view-projected
    // fit around the *focus* center but over the scene's full extent gives the
    // distance needed to clear all of it; when focus == scene (the common
    // whole-structure case) the two radii coincide, so this never zooms out
    // beyond what ferritin-ala.5 already tightened.
    if let Some((scene_min, scene_max)) = ctx.scene_bounds {
        let scene_radius = fit_radius_for_view_from_center(
            ctx.orbit.focus,
            scene_min,
            scene_max,
            ctx.orbit,
            ctx.aspect_ratio,
        );
        radius = radius.max(scene_radius);
    }

    ctx.orbit.radius = radius;
}

/// Transform an axis-aligned local `[min, max]` box into world space, accounting
/// for rotation by transforming all 8 corners rather than just the two extremes
/// (a rotation can change which corner is extremal on a given axis).
fn transform_aabb(local_min: Vec3, local_max: Vec3, transform: Transform) -> (Vec3, Vec3) {
    let mut world_min = Vec3::splat(f32::MAX);
    let mut world_max = Vec3::splat(f32::MIN);
    for &x in &[local_min.x, local_max.x] {
        for &y in &[local_min.y, local_max.y] {
            for &z in &[local_min.z, local_max.z] {
                let corner = transform.transform_point(Vec3::new(x, y, z));
                world_min = world_min.min(corner);
                world_max = world_max.max(corner);
            }
        }
    }
    (world_min, world_max)
}

/// Camera distance that keeps the AABB `[min, max]` fully in view for the current
/// orbit yaw/pitch/up, centered on the AABB's own midpoint.
///
/// Using the worst-case any-angle bounding sphere (`aabb_diagonal / 2`) over-estimates
/// how far back the camera needs to be for any real (non-spherical) structure, since
/// the 3D corner-to-corner diagonal is longer than the object's extent along any single
/// screen axis — this left single structures filling only a small fraction of the
/// viewport (ferritin-ala.5). Projecting the AABB corners onto the camera's actual
/// right/up axes gives a much tighter fit for the current view direction.
fn fit_radius_for_view(min: Vec3, max: Vec3, orbit: &OrbitCamera, aspect: f32) -> f32 {
    let center = (min + max) * 0.5;
    fit_radius_for_view_from_center(center, min, max, orbit, aspect)
}

/// As [`fit_radius_for_view`], but projects the AABB corners relative to an
/// explicit `center` rather than the AABB's own midpoint. Used to ask "how far
/// back must the camera sit, still looking at `center`, to keep this other AABB
/// in view" — e.g. clearing the whole scene while focused on a small target
/// within it (ferritin-t0h.5).
///
/// `aspect` (viewport width / height) determines the true horizontal FOV; using
/// the vertical FOV for both axes over-estimated the distance needed to fit the
/// horizontal extent on a widescreen viewport, leaving whole-structure framing
/// filling under half the frame (ferritin-t0h.10).
fn fit_radius_for_view_from_center(
    center: Vec3,
    min: Vec3,
    max: Vec3,
    orbit: &OrbitCamera,
    aspect: f32,
) -> f32 {
    // Points from focus toward the camera (same convention as `orbit_position`'s
    // offset); only its axis matters for building a look-at basis, not its sign.
    let view_axis = Vec3::new(
        orbit.yaw.sin() * orbit.pitch.cos(),
        orbit.pitch.sin(),
        orbit.yaw.cos() * orbit.pitch.cos(),
    );
    let world_up = if orbit.up.length_squared() > 1e-6 {
        orbit.up
    } else {
        Vec3::Y
    };
    let right = view_axis.cross(world_up).normalize_or_zero();
    let cam_up = right.cross(view_axis).normalize_or_zero();

    let mut half_right = 0.0_f32;
    let mut half_up = 0.0_f32;
    for &x in &[min.x, max.x] {
        for &y in &[min.y, max.y] {
            for &z in &[min.z, max.z] {
                let corner = Vec3::new(x, y, z) - center;
                half_right = half_right.max(corner.dot(right).abs());
                half_up = half_up.max(corner.dot(cam_up).abs());
            }
        }
    }

    // Half of Bevy's default *vertical* FOV (45°), used by every Camera3d in this
    // codebase. The horizontal half-FOV is derived from it via the viewport
    // aspect ratio (standard perspective-projection relation), not assumed equal
    // to the vertical one (ferritin-t0h.10).
    const V_HALF_FOV: f32 = std::f32::consts::FRAC_PI_4 / 2.0;
    let h_half_fov = (aspect * V_HALF_FOV.tan()).atan();
    const MARGIN: f32 = 1.15;
    let radius_for_up = half_up / V_HALF_FOV.tan();
    let radius_for_right = half_right / h_half_fov.tan();
    (radius_for_up.max(radius_for_right) * MARGIN).max(1.0)
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
        RepresentationTypeT::BallAndStick => (RenderOptions::BallAndStick, false),
        RepresentationTypeT::Cartoon => (RenderOptions::Cartoon, false),
        RepresentationTypeT::Line => (RenderOptions::Wireframe, false),
        RepresentationTypeT::Putty => (RenderOptions::Putty, false),
        RepresentationTypeT::Spacefill => (RenderOptions::Solid, false),
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
/// Axis-aligned bounding box (min, max) of the masked atoms; `None` when no atom passes.
fn aabb_of_masked(ac: &AtomCollection, mask: &AtomMask) -> Option<(Vec3, Vec3)> {
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
    Some((min, max))
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
            .query_filtered::<Entity, (With<Mesh3d>, With<MvsEntity>, Without<MvsRetiredEntity>)>()
            .iter(app.world())
            .count()
    }

    fn count_retired_meshes(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<Entity, (With<Mesh3d>, With<MvsRetiredEntity>)>()
            .iter(app.world())
            .count()
    }

    fn count_labels(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<Entity, (With<MvsLabel>, Without<MvsRetiredEntity>)>()
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
        let (min, max) = aabb_of_masked(&ac, &mask).unwrap();
        let center = (min + max) * 0.5;
        let diag = (max - min).length();
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
    fn test_map_repr_spacefill() {
        let (opt, degraded) = map_representation(&RepresentationTypeT::Spacefill);
        assert!(matches!(opt, RenderOptions::Solid));
        assert!(
            !degraded,
            "Spacefill maps to Solid with VDW radii — not degraded"
        );
    }

    #[test]
    fn test_map_repr_putty() {
        let (opt, degraded) = map_representation(&RepresentationTypeT::Putty);
        assert!(matches!(opt, RenderOptions::Putty));
        assert!(!degraded);
    }

    #[test]
    fn test_map_repr_line() {
        let (opt, degraded) = map_representation(&RepresentationTypeT::Line);
        assert!(matches!(opt, RenderOptions::Wireframe));
        assert!(!degraded);
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

    // ferritin-ala.3: a label's world anchor gets a companion billboard Text/UiNode
    // once a Camera3d exists, positioned via the camera's viewport projection.
    #[test]
    fn test_label_billboard_ui_node_created() {
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
        let camera_transform = Transform::from_xyz(0.0, 0.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y);
        let viewport_size = UVec2::new(800, 600);
        let mut projection = Projection::default();
        projection.update(viewport_size.x as f32, viewport_size.y as f32);
        let mut camera = Camera {
            viewport: Some(bevy::camera::Viewport {
                physical_size: viewport_size,
                ..default()
            }),
            ..default()
        };
        // MinimalPlugins doesn't run the camera-projection update system that
        // normally populates `Camera::computed`, so seed it manually (mirrors
        // bevy_camera's own internal `make_camera` test helper).
        camera.computed.clip_from_view = projection.get_clip_from_view();
        camera.computed.target_info = Some(bevy::camera::RenderTargetInfo {
            physical_size: viewport_size,
            scale_factor: 1.0,
        });
        app.world_mut().spawn((
            Camera3d::default(),
            camera,
            projection,
            camera_transform,
            GlobalTransform::from(camera_transform),
        ));
        send_and_update(&mut app, &json);
        // execute_mvs_on_load and update_mvs_label_billboards are `.chain()`d, so a
        // single update() already runs both; run one more to let GlobalTransform
        // propagation (if any plugin performs it) settle before re-checking.
        app.update();

        let mut label_query = app.world_mut().query::<&MvsLabelUiNode>();
        assert_eq!(
            label_query.iter(app.world()).count(),
            1,
            "expected the label anchor to gain a MvsLabelUiNode link"
        );

        let mut text_query = app.world_mut().query::<(&Text, &UiNode)>();
        let (text, node) = text_query
            .single(app.world())
            .expect("expected exactly one billboard Text/UiNode entity");
        assert_eq!(text.0, "Hello World");
        assert_eq!(node.position_type, PositionType::Absolute);
    }

    // ferritin-t0h.9: Line/Wireframe geometry is now thin triangle cylinders with
    // real normals (no longer a normal-less GPU LineList), so it renders lit like
    // every other representation instead of being forced unlit.
    #[test]
    fn test_line_representation_material_is_lit() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(&mut app, &mvsj_one_component(&path, "all", "line"));

        let mut query = app.world_mut().query::<&MeshMaterial3d<StandardMaterial>>();
        let handle = query
            .single(app.world())
            .expect("expected exactly one mesh material")
            .clone();
        let materials = app.world().resource::<Assets<StandardMaterial>>();
        let material = materials.get(&handle.0).expect("material must exist");
        assert!(
            !material.unlit,
            "line representation material should keep normal PBR shading"
        );
    }

    #[test]
    fn test_ball_and_stick_material_is_lit() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(
            &mut app,
            &mvsj_one_component(&path, "all", "ball_and_stick"),
        );

        let mut query = app.world_mut().query::<&MeshMaterial3d<StandardMaterial>>();
        let handle = query
            .single(app.world())
            .expect("expected exactly one mesh material")
            .clone();
        let materials = app.world().resource::<Assets<StandardMaterial>>();
        let material = materials.get(&handle.0).expect("material must exist");
        assert!(
            !material.unlit,
            "ball_and_stick representation material should keep normal PBR shading"
        );
    }

    // ferritin-ala.7: Surface silently degrades to VdW spheres (Solid) with only a
    // log warning -- the viewer itself gave no visual signal that it wasn't a real
    // surface render, pixel-identical to Spacefill. It should now be tinted.
    #[test]
    fn test_surface_fallback_is_tinted() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(&mut app, &mvsj_one_component(&path, "all", "surface"));

        let mut query = app.world_mut().query::<&MeshMaterial3d<StandardMaterial>>();
        let handle = query
            .single(app.world())
            .expect("expected exactly one mesh material")
            .clone();
        let materials = app.world().resource::<Assets<StandardMaterial>>();
        let material = materials.get(&handle.0).expect("material must exist");
        assert_ne!(
            material.base_color,
            Color::WHITE,
            "degraded Surface fallback should be visibly tinted, not default white"
        );
    }

    // ferritin-ala.9: the Surface degradation must reach the MvsError message
    // channel (which a status bar subscribes to — see molviewspec_viewer.rs's
    // collect_errors/format_error), not just a log-only warn!() call.
    #[test]
    fn test_surface_fallback_emits_mvs_error() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(&mut app, &mvsj_one_component(&path, "all", "surface"));

        let errors = get_errors(&app);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                MvsError::RepresentationDegraded {
                    requested: RepresentationTypeT::Surface,
                    ..
                }
            )),
            "expected a RepresentationDegraded error for the Surface fallback, got {errors:?}"
        );
    }

    #[test]
    fn test_spacefill_is_not_tinted() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();
        let mut app = make_app();
        send_and_update(&mut app, &mvsj_one_component(&path, "all", "spacefill"));

        let mut query = app.world_mut().query::<&MeshMaterial3d<StandardMaterial>>();
        let handle = query
            .single(app.world())
            .expect("expected exactly one mesh material")
            .clone();
        let materials = app.world().resource::<Assets<StandardMaterial>>();
        let material = materials.get(&handle.0).expect("material must exist");
        assert_eq!(
            material.base_color,
            Color::WHITE,
            "a real (non-degraded) spacefill request should not carry the fallback tint"
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

    // ferritin-t0h.5: focusing a tiny target (e.g. a single residue, standing in
    // for "two ions") that sits inside a much larger *already-rendered* structure
    // must not zoom the camera in tight enough to near-clip through the
    // surrounding geometry. Compares the same tiny focus target in isolation
    // (nothing else rendered) against the same target focused alongside a full
    // cartoon of the whole protein: the latter's radius must be pulled back
    // well beyond what the tiny AABB alone would ever justify.
    #[test]
    fn test_focus_small_target_clamped_to_scene_bounds() {
        use ferritin_test_data::TestFile;

        let (path, _tmp) = TestFile::protein_01().create_temp().unwrap();

        // Scene A: only the tiny target (first residue) is rendered and focused.
        let json_small_only = format!(
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
                        "params": {{"selector": {{"residue_index": 0}}}},
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
        let mut app_small = make_app();
        send_and_update(&mut app_small, &json_small_only);
        let radius_small_only = app_small.world().resource::<OrbitCamera>().radius;

        // Scene B: the whole protein is rendered (cartoon, unfocused) first, then
        // the same tiny residue is focused — mirroring preset_label's structure
        // (full protein cartoon, then a small focused ion/label target).
        let json_with_scene = format!(
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
                          "params": {{"selector": "all"}},
                          "children": [
                            {{"kind": "representation", "params": {{"type": "cartoon"}}}}
                          ]
                        }},
                        {{
                          "kind": "component",
                          "params": {{"selector": {{"residue_index": 0}}}},
                          "children": [
                            {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}},
                            {{"kind": "focus", "params": {{}}}}
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
        let mut app_scene = make_app();
        send_and_update(&mut app_scene, &json_with_scene);
        let radius_with_scene = app_scene.world().resource::<OrbitCamera>().radius;

        assert!(
            radius_with_scene > radius_small_only * 3.0,
            "expected the scene-bounds clamp to pull the camera back well beyond \
             the tiny target's own fit (small-only={radius_small_only}, \
             with-scene={radius_with_scene})"
        );
    }

    // ferritin-t0h.10: on a widescreen viewport, a horizontally-dominant AABB
    // should need a *smaller* radius once the true (wider) horizontal FOV is
    // used, instead of being fit as if the horizontal FOV equalled the narrower
    // vertical one. A square aspect ratio (1:1) is the boundary case where
    // horizontal and vertical FOV coincide, so it acts as the "old behavior"
    // reference point here.
    #[test]
    fn test_fit_radius_tighter_on_widescreen_for_wide_aabb() {
        let orbit = OrbitCamera::default();
        // Wide, flat AABB: extent along the camera's right axis dominates.
        let min = Vec3::new(-100.0, -5.0, -5.0);
        let max = Vec3::new(100.0, 5.0, 5.0);

        let radius_square = fit_radius_for_view(min, max, &orbit, 1.0);
        let radius_widescreen = fit_radius_for_view(min, max, &orbit, 16.0 / 9.0);

        assert!(
            radius_widescreen < radius_square,
            "widescreen fit ({radius_widescreen}) should pull in tighter than \
             square fit ({radius_square}) for a horizontally-dominant AABB"
        );
    }

    // ferritin-t0h.10 (contrast case): a vertically-dominant AABB is bounded by
    // the *vertical* FOV regardless of aspect ratio, so widening the viewport
    // must not change the fitted radius at all.
    #[test]
    fn test_fit_radius_unaffected_by_aspect_for_tall_aabb() {
        let orbit = OrbitCamera::default();
        let min = Vec3::new(-5.0, -100.0, -5.0);
        let max = Vec3::new(5.0, 100.0, 5.0);

        let radius_square = fit_radius_for_view(min, max, &orbit, 1.0);
        let radius_widescreen = fit_radius_for_view(min, max, &orbit, 16.0 / 9.0);

        assert!(
            (radius_widescreen - radius_square).abs() < 1e-3,
            "vertical-extent fit should be aspect-independent: square={radius_square}, \
             widescreen={radius_widescreen}"
        );
    }

    // ferritin-ala.5: focus() should fill most of the viewport, not just a small
    // fraction of it. Reconstructs the same camera the viewer examples use
    // (Camera3d::default(), orbit_position formula) and projects the real rendered
    // mesh's vertices to check on-screen coverage, rather than trusting the
    // orbit-radius formula's own math (which is exactly what was wrong before).
    #[test]
    fn test_focus_fills_most_of_viewport() {
        use ferritin_test_data::TestFile;
        let (path, _tmp) = TestFile::mvs_4hhb().create_temp().unwrap();
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
                          {{"kind": "representation", "params": {{"type": "cartoon"}}}},
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
        send_and_update(&mut app, &json);
        let orbit = app.world().resource::<OrbitCamera>().clone();

        let camera_transform = Transform::from_translation({
            let x = orbit.radius * orbit.yaw.sin() * orbit.pitch.cos();
            let y = orbit.radius * orbit.pitch.sin();
            let z = orbit.radius * orbit.yaw.cos() * orbit.pitch.cos();
            orbit.focus + Vec3::new(x, y, z)
        })
        .looking_at(orbit.focus, Vec3::Y);
        let viewport_size = UVec2::new(2560, 1440);
        let mut projection = Projection::default();
        projection.update(viewport_size.x as f32, viewport_size.y as f32);
        let mut camera = Camera {
            viewport: Some(bevy::camera::Viewport {
                physical_size: viewport_size,
                ..default()
            }),
            ..default()
        };
        camera.computed.clip_from_view = projection.get_clip_from_view();
        camera.computed.target_info = Some(bevy::camera::RenderTargetInfo {
            physical_size: viewport_size,
            scale_factor: 1.0,
        });
        let gt = GlobalTransform::from(camera_transform);

        let mut mesh_query = app.world_mut().query::<&Mesh3d>();
        let mesh_handle = mesh_query
            .single(app.world())
            .expect("expected 1 mesh")
            .clone();
        let meshes = app.world().resource::<Assets<Mesh>>();
        let mesh = meshes.get(&mesh_handle.0).expect("mesh must exist");
        let positions = match mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .expect("mesh must have positions")
        {
            bevy::render::mesh::VertexAttributeValues::Float32x3(v) => v,
            _ => panic!("unexpected position format"),
        };

        let mut min_px = Vec2::splat(f32::MAX);
        let mut max_px = Vec2::splat(f32::MIN);
        for p in positions {
            if let Ok(px) = camera.world_to_viewport(&gt, Vec3::from_array(*p)) {
                min_px = min_px.min(px);
                max_px = max_px.max(px);
            }
        }

        let fill_height = (max_px.y - min_px.y) / viewport_size.y as f32;
        let fill_width = (max_px.x - min_px.x) / viewport_size.x as f32;
        assert!(
            fill_height > 0.5 || fill_width > 0.5,
            "expected focus() to fill most of the viewport, got fill_height={fill_height:.2} fill_width={fill_width:.2}"
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

    // ferritin-ala.8/ferritin-c1k.4: MvsPlugin should install molecular-viewer
    // camera defaults for color fidelity and depth cues.
    #[test]
    fn test_camera3d_gets_molecular_viewer_defaults() {
        let mut app = make_app();
        let camera = app
            .world_mut()
            .spawn((Camera3d::default(), Transform::default()))
            .id();
        app.update();

        let tonemapping = app.world().get::<Tonemapping>(camera);
        assert_eq!(
            tonemapping,
            Some(&Tonemapping::None),
            "Camera3d should have tonemapping disabled by MvsPlugin"
        );
        let ssao = app.world().get::<ScreenSpaceAmbientOcclusion>(camera);
        assert_eq!(
            ssao.map(|s| s.quality_level),
            Some(ScreenSpaceAmbientOcclusionQualityLevel::High),
            "Camera3d should have SSAO enabled by MvsPlugin"
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
        // Active scene count stays at 1; the old mesh remains hidden briefly to
        // avoid same-frame render-allocator free/reallocate churn.
        assert_eq!(
            count_meshes(&mut app),
            1,
            "T3-23: second load replaces first, still 1 mesh"
        );
        assert_eq!(
            count_retired_meshes(&mut app),
            1,
            "T3-23: old scene is retired for one frame"
        );

        app.update();
        assert_eq!(
            count_retired_meshes(&mut app),
            0,
            "T3-23: retired scene is cleaned up on the following update"
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

    // ferritin-ala.5: a superposition-style scene with two focus()ed structures
    // should frame the union of both, not just the last one processed (which
    // previously clipped the first/earlier structure out of frame).
    #[test]
    fn test_two_focused_structures_frame_union() {
        use ferritin_test_data::TestFile;
        let (path1, _tmp1) = TestFile::protein_01().create_temp().unwrap();
        let (path2, _tmp2) = TestFile::protein_01().create_temp().unwrap();
        // Second copy translated 300 units along X: if only the last focus() won,
        // orbit.focus.x would land near 300, not near the midpoint (~150).
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
                          "children": [
                            {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}},
                            {{"kind": "focus", "params": {{}}}}
                          ]
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
                        "children": [
                          {{
                            "kind": "transform",
                            "params": {{
                              "rotation": [1,0,0, 0,1,0, 0,0,1],
                              "translation": [300.0, 0.0, 0.0]
                            }}
                          }},
                          {{
                            "kind": "component",
                            "params": {{"selector": "all"}},
                            "children": [
                              {{"kind": "representation", "params": {{"type": "ball_and_stick"}}}},
                              {{"kind": "focus", "params": {{}}}}
                            ]
                          }}
                        ]
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
        assert!(
            orbit.focus.x > 50.0 && orbit.focus.x < 250.0,
            "expected focus to land near the midpoint of both structures (~150), got {}",
            orbit.focus.x
        );
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
