//! Interactive MolViewSpec (MVS) viewer.
//!
//! Loads MolViewSpec states and renders them with the ferritin-bevy [`MvsPlugin`]
//! execution engine. The UI (styled with `bevy_feathers` dark theme tokens) has:
//!
//!   - a top bar: title, an editable `.mvsj` file-path field, and a Load button
//!   - a row of five preset buttons that build MVS states from local
//!     `ferritin-test-data` structures (no HTTP/HTTPS)
//!   - a left sidebar: a collapsible inspector of the loaded MVS node tree
//!   - the main 3D viewport (orbit camera shared with the executor)
//!   - a bottom status bar: `MvsError` events in red, "Ready" otherwise
//!
//! Camera controls (when the path field is not focused):
//!   Left-drag — orbit    Right-drag — pan    Scroll — zoom    R — reset
//!
//! NOTE on icons: `bevy_feathers` 0.19 ships only chevron/x icons (no
//! folder/refresh/camera-reset), so all buttons here are text-only. See bead
//! `ferritin-bevy-feathers-icons-2026-06-30`.
//!
//! Run with:
//!   cargo run --example bevy_molviewspec_viewer --features bevy -p ferritin-examples

use anyhow::Result;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::input::ButtonState;
use bevy::prelude::*;
use bevy_feathers::{
    dark_theme::create_dark_theme,
    theme::{ThemeBackgroundColor, UiTheme},
    tokens, FeathersPlugins,
};
use ferritin_bevy::{LoadMvsEvent, MvsError, MvsPlugin, MvsStateResource, OrbitCamera};
use ferritin_molviewspec::molviewspec::nodes::{
    ColorNamesT, ColorT, ComponentExpression, ComponentSelector, ComponentSelectorT, KindT,
    Node as MvsNode, NodeParams, ParseFormatT, ParseParams, RepresentationTypeT, State,
    StructureParams, StructureTypeT,
};
use ferritin_test_data::TestFile;
use std::path::PathBuf;
use tempfile::NamedTempFile;

// ---- Colour constants for manual button styling --------------------------------

const BTN_NORMAL: Color = Color::srgb(0.18, 0.18, 0.20);
const BTN_HOVER: Color = Color::srgb(0.28, 0.28, 0.32);
const BTN_ACTIVE: Color = Color::srgb(0.22, 0.42, 0.64);
const TEXT_DIM: Color = Color::srgb(0.75, 0.75, 0.75);
const TEXT_BRIGHT: Color = Color::srgb(0.92, 0.92, 0.92);
const STATUS_OK: Color = Color::srgb(0.55, 0.85, 0.55);
const STATUS_ERR: Color = Color::srgb(0.95, 0.45, 0.45);

// ---- Resources -----------------------------------------------------------------

/// Each preset's display name and the pre-built MVSJ document it loads.
#[derive(Resource)]
struct Presets(Vec<Preset>);

struct Preset {
    name: &'static str,
    json: String,
}

/// Keeps the extracted `.cif` temp files alive for the whole run; the MVS
/// `download` nodes reference them by path.
#[derive(Resource)]
#[allow(dead_code)]
struct PresetTempFiles(Vec<NamedTempFile>);

/// Editable text of the top-bar file-path field, plus whether it has focus.
#[derive(Resource, Default)]
struct PathInput {
    text: String,
    focused: bool,
}

/// Current status-bar contents.
#[derive(Resource)]
struct StatusBar {
    text: String,
    error: bool,
}

impl Default for StatusBar {
    fn default() -> Self {
        Self {
            text: "Ready — pick a preset or enter a .mvsj path.".to_string(),
            error: false,
        }
    }
}

/// Node tree paths (vectors of child indices) that are collapsed in the inspector.
#[derive(Resource, Default)]
struct TreeCollapse(std::collections::HashSet<Vec<usize>>);

/// Set when the inspector needs to be rebuilt (after a load or a collapse toggle).
#[derive(Resource)]
struct TreeDirty(bool);

// ---- Components ----------------------------------------------------------------

#[derive(Component, Clone)]
struct PresetButton(usize);

#[derive(Component)]
struct LoadButton;

#[derive(Component)]
struct PathFieldButton;

#[derive(Component)]
struct PathText;

#[derive(Component)]
struct StatusText;

#[derive(Component)]
struct TreePanel;

/// A single inspector row; carries the node-tree path it represents so a click
/// can toggle that node's collapse state.
#[derive(Component)]
struct TreeRow(Vec<usize>);

// ---- Entry point ---------------------------------------------------------------

fn main() -> Result<()> {
    // Extract the preset structures to temp files and build their MVS documents.
    let (presets, temp_files) = build_presets()?;

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FeathersPlugins)
        .add_plugins(MvsPlugin)
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(Presets(presets))
        .insert_resource(PresetTempFiles(temp_files))
        .insert_resource(PathInput::default())
        .insert_resource(StatusBar::default())
        .insert_resource(TreeCollapse::default())
        .insert_resource(TreeDirty(false))
        .add_systems(Startup, (setup_scene, setup_ui))
        .add_systems(
            Update,
            (
                handle_preset_buttons,
                handle_load_button,
                handle_path_field_focus,
                edit_path_field,
                update_path_text,
                collect_errors,
                mark_tree_dirty_on_load,
                rebuild_tree,
                handle_tree_rows,
                render_status,
                autoframe_on_load,
                orbit_input,
                sync_camera,
                button_hover_colors,
            ),
        )
        .run();

    Ok(())
}

// ---- Preset construction -------------------------------------------------------

/// Extract every preset structure to a temp `.cif`, then build the five MVS
/// documents that reference them by local path.
fn build_presets() -> Result<(Vec<Preset>, Vec<NamedTempFile>)> {
    let mut temps = Vec::new();

    // Helper: write a TestFile to a temp file, return its path, keep the handle.
    let mut extract = |tf: TestFile| -> Result<String> {
        let (path, handle) = tf.create_temp()?;
        temps.push(handle);
        Ok(path)
    };

    let p_1cbs = extract(TestFile::mvs_1cbs())?;
    let p_1c0a = extract(TestFile::mvs_1c0a())?;
    let p_1lap = extract(TestFile::mvs_1lap())?;
    let p_4hhb = extract(TestFile::mvs_4hhb())?;
    let p_1oj6 = extract(TestFile::mvs_1oj6())?;
    let p_1tqn = extract(TestFile::mvs_1tqn())?;

    let presets = vec![
        Preset {
            name: "Basic",
            json: preset_basic(&p_1cbs),
        },
        Preset {
            name: "Components",
            json: preset_components(&p_1c0a),
        },
        Preset {
            name: "Label",
            json: preset_label(&p_1lap),
        },
        Preset {
            name: "Superposition",
            json: preset_superposition(&p_4hhb, &p_1oj6),
        },
        Preset {
            name: "Symmetry",
            json: preset_symmetry(&p_1tqn),
        },
    ];

    Ok((presets, temps))
}

fn mmcif() -> ParseParams {
    ParseParams {
        format: ParseFormatT::Mmcif,
    }
}

fn model_params() -> StructureParams {
    StructureParams {
        structure_type: StructureTypeT::Model,
        ..Default::default()
    }
}

fn sel(s: ComponentSelectorT) -> ComponentSelector {
    ComponentSelector::Selector(s)
}

fn expr(e: ComponentExpression) -> ComponentSelector {
    ComponentSelector::Expression(e)
}

/// Basic: 1cbs whole structure as cartoon coloured blue, framed by a focus node.
fn preset_basic(path: &str) -> String {
    let mut state = State::new();
    {
        let structure = state
            .download(path)
            .unwrap()
            .parse(mmcif())
            .unwrap()
            .model_structure(model_params())
            .unwrap();
        let comp = structure.component(sel(ComponentSelectorT::All)).unwrap();
        comp.representation(RepresentationTypeT::Cartoon)
            .unwrap()
            .color(
                ColorT::Named(ColorNamesT::Blue),
                sel(ComponentSelectorT::All),
            );
        comp.focus(None, None);
    }
    to_json(&state)
}

/// Components: 1c0a with protein / nucleic / ligand as separate coloured
/// representations, plus two highlighted arginine residues with labels.
fn preset_components(path: &str) -> String {
    let orange = ColorT::Hex("#e19039".to_string());
    let blue = ColorT::Hex("#4b7fcc".to_string());
    let green = ColorT::Hex("#229954".to_string());
    let red = ColorT::Hex("#ff0000".to_string());

    let mut state = State::new();
    {
        let structure = state
            .download(path)
            .unwrap()
            .parse(mmcif())
            .unwrap()
            .model_structure(model_params())
            .unwrap();

        {
            let c = structure.component(sel(ComponentSelectorT::Protein)).unwrap();
            c.representation(RepresentationTypeT::Cartoon)
                .unwrap()
                .color(orange, sel(ComponentSelectorT::All));
        }
        {
            let c = structure.component(sel(ComponentSelectorT::Nucleic)).unwrap();
            c.representation(RepresentationTypeT::Cartoon)
                .unwrap()
                .color(blue, sel(ComponentSelectorT::All));
        }
        {
            let c = structure.component(sel(ComponentSelectorT::Ligand)).unwrap();
            c.representation(RepresentationTypeT::BallAndStick)
                .unwrap()
                .color(green, sel(ComponentSelectorT::All));
        }
        for seq in [217, 537] {
            let residue = ComponentExpression {
                auth_asym_id: Some("B".to_string()),
                auth_seq_id: Some(seq),
                ..Default::default()
            };
            let c = structure.component(expr(residue)).unwrap();
            c.representation(RepresentationTypeT::BallAndStick)
                .unwrap()
                .color(red.clone(), sel(ComponentSelectorT::All));
            c.label("aaRS Class II Signature".to_string());
        }
        {
            // Frame the whole structure.
            let c = structure.component(sel(ComponentSelectorT::All)).unwrap();
            c.focus(None, None);
        }
    }
    to_json(&state)
}

/// Label: 1lap whole structure as cartoon, with the catalytic zinc ions shown as
/// labelled ball-and-stick and the camera focused on them.
fn preset_label(path: &str) -> String {
    let mut state = State::new();
    {
        let structure = state
            .download(path)
            .unwrap()
            .parse(mmcif())
            .unwrap()
            .model_structure(model_params())
            .unwrap();
        {
            let c = structure.component(sel(ComponentSelectorT::Protein)).unwrap();
            c.representation(RepresentationTypeT::Cartoon).unwrap().color(
                ColorT::Named(ColorNamesT::Lightgray),
                sel(ComponentSelectorT::All),
            );
        }
        {
            let c = structure.component(sel(ComponentSelectorT::Ion)).unwrap();
            c.representation(RepresentationTypeT::BallAndStick)
                .unwrap()
                .color(
                    ColorT::Named(ColorNamesT::Magenta),
                    sel(ComponentSelectorT::All),
                );
            c.label("Catalytic metal site".to_string());
            c.focus(None, None);
        }
    }
    to_json(&state)
}

/// Superposition: two structures in one scene; the second is offset by a
/// translation transform so both are visible side by side.
fn preset_superposition(path_a: &str, path_b: &str) -> String {
    let mut state = State::new();
    {
        let structure = state
            .download(path_a)
            .unwrap()
            .parse(mmcif())
            .unwrap()
            .model_structure(model_params())
            .unwrap();
        let c = structure.component(sel(ComponentSelectorT::All)).unwrap();
        c.representation(RepresentationTypeT::Cartoon).unwrap().color(
            ColorT::Named(ColorNamesT::Steelblue),
            sel(ComponentSelectorT::All),
        );
        c.focus(None, None);
    }
    {
        use ferritin_molviewspec::molviewspec::nodes::TransformParams;
        let structure = state
            .download(path_b)
            .unwrap()
            .parse(mmcif())
            .unwrap()
            .model_structure(model_params())
            .unwrap();
        structure.transform(TransformParams {
            rotation: None,
            translation: Some((80.0, 0.0, 0.0)),
        });
        let c = structure.component(sel(ComponentSelectorT::All)).unwrap();
        c.representation(RepresentationTypeT::Cartoon).unwrap().color(
            ColorT::Named(ColorNamesT::Orange),
            sel(ComponentSelectorT::All),
        );
    }
    to_json(&state)
}

/// Symmetry: 1tqn requested as a symmetry structure. Assembly/symmetry expansion
/// is unsupported, so the executor degrades gracefully (status bar shows a
/// warning) and renders the deposited model.
fn preset_symmetry(path: &str) -> String {
    let mut state = State::new();
    {
        let params = StructureParams {
            structure_type: StructureTypeT::Symmetry,
            ..Default::default()
        };
        let structure = state
            .download(path)
            .unwrap()
            .parse(mmcif())
            .unwrap()
            .symmetry_structure(params)
            .unwrap();
        let c = structure.component(sel(ComponentSelectorT::All)).unwrap();
        c.representation(RepresentationTypeT::Cartoon).unwrap().color(
            ColorT::Named(ColorNamesT::Seagreen),
            sel(ComponentSelectorT::All),
        );
        c.focus(None, None);
    }
    to_json(&state)
}

fn to_json(state: &State) -> String {
    serde_json::to_string(state).expect("MVS state serializes to JSON")
}

// ---- 3-D scene -----------------------------------------------------------------

fn setup_scene(mut commands: Commands, orbit: Res<OrbitCamera>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(orbit_position(&orbit)).looking_at(orbit.focus, orbit.up),
    ));

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.9, 0.9),
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.8, 0.8, 1.0),
            illuminance: 5_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.5, -0.5, 0.0)),
    ));
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.9, 0.9, 1.0),
            illuminance: 3_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.0, std::f32::consts::PI, 0.0)),
    ));
}

/// World-space camera position from the orbit parameters (matches the convention
/// used by the executor's `set_yaw_pitch_from_offset`).
fn orbit_position(orbit: &OrbitCamera) -> Vec3 {
    let x = orbit.radius * orbit.yaw.sin() * orbit.pitch.cos();
    let y = orbit.radius * orbit.pitch.sin();
    let z = orbit.radius * orbit.yaw.cos() * orbit.pitch.cos();
    orbit.focus + Vec3::new(x, y, z)
}

// ---- Camera systems ------------------------------------------------------------

fn orbit_input(
    mut orbit: ResMut<OrbitCamera>,
    camera: Query<&Transform, With<Camera3d>>,
    path_input: Res<PathInput>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
) {
    let delta = motion.delta;
    let scroll_delta = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y * 5.0,
        MouseScrollUnit::Pixel => scroll.delta.y * 0.1,
    };

    if mouse.pressed(MouseButton::Left) && delta != Vec2::ZERO {
        orbit.yaw -= delta.x * 0.005;
        orbit.pitch = (orbit.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
    }

    if mouse.pressed(MouseButton::Right) && delta != Vec2::ZERO {
        if let Ok(transform) = camera.single() {
            let right = transform.right();
            let up = transform.up();
            let pan_scale = orbit.radius * 0.001;
            orbit.focus += (right * -delta.x + up * delta.y) * pan_scale;
        }
    }

    if scroll_delta != 0.0 {
        orbit.radius = (orbit.radius - scroll_delta).max(1.0);
    }

    // R resets the camera, but only when the user is not typing in the path field.
    if !path_input.focused && keys.just_pressed(KeyCode::KeyR) {
        *orbit = OrbitCamera::default();
    }
}

fn sync_camera(orbit: Res<OrbitCamera>, mut camera: Query<&mut Transform, With<Camera3d>>) {
    if let Ok(mut transform) = camera.single_mut() {
        transform.translation = orbit_position(&orbit);
        let up = if orbit.up.length_squared() > 1e-6 {
            orbit.up
        } else {
            Vec3::Y
        };
        transform.look_at(orbit.focus, up);
    }
}

/// After a load, if the loaded state carries no `camera`/`focus` node to frame the
/// scene, reset the orbit camera to its default position so the structure (which
/// the executor renders at its original coordinates) is at least in view.
fn autoframe_on_load(state_res: Res<MvsStateResource>, mut orbit: ResMut<OrbitCamera>) {
    if !state_res.is_changed() {
        return;
    }
    let Some(state) = &state_res.0 else {
        return;
    };
    if tree_has_camera_or_focus(&state.root) {
        return; // explicit camera/focus framing wins.
    }
    *orbit = OrbitCamera::default();
}

fn tree_has_camera_or_focus(node: &MvsNode) -> bool {
    if matches!(node.kind, KindT::Camera | KindT::Focus) {
        return true;
    }
    node.children
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .any(tree_has_camera_or_focus)
}

// ---- Button / load handling ----------------------------------------------------

fn handle_preset_buttons(
    query: Query<(&Interaction, &PresetButton), Changed<Interaction>>,
    presets: Res<Presets>,
    mut loader: MessageWriter<LoadMvsEvent>,
    mut status: ResMut<StatusBar>,
) {
    for (interaction, button) in &query {
        if *interaction == Interaction::Pressed {
            if let Some(preset) = presets.0.get(button.0) {
                loader.write(LoadMvsEvent::FromString(preset.json.clone()));
                status.text = format!("Loaded preset: {}", preset.name);
                status.error = false;
            }
        }
    }
}

fn handle_load_button(
    query: Query<&Interaction, (Changed<Interaction>, With<LoadButton>)>,
    path_input: Res<PathInput>,
    mut loader: MessageWriter<LoadMvsEvent>,
    mut status: ResMut<StatusBar>,
) {
    for interaction in &query {
        if *interaction == Interaction::Pressed {
            let path = path_input.text.trim();
            if path.is_empty() {
                status.text = "Enter a .mvsj file path first.".to_string();
                status.error = true;
            } else {
                loader.write(LoadMvsEvent::FromFile(PathBuf::from(path)));
                status.text = format!("Loading {path} ...");
                status.error = false;
            }
        }
    }
}

fn handle_path_field_focus(
    query: Query<&Interaction, (Changed<Interaction>, With<PathFieldButton>)>,
    presets: Query<&Interaction, (Changed<Interaction>, With<PresetButton>)>,
    mut path_input: ResMut<PathInput>,
) {
    for interaction in &query {
        if *interaction == Interaction::Pressed {
            path_input.focused = true;
        }
    }
    // Clicking a preset takes focus away from the text field.
    for interaction in &presets {
        if *interaction == Interaction::Pressed {
            path_input.focused = false;
        }
    }
}

fn edit_path_field(mut input: ResMut<PathInput>, mut keys: MessageReader<KeyboardInput>) {
    if !input.focused {
        keys.clear();
        return;
    }
    for ev in keys.read() {
        if ev.state != ButtonState::Pressed {
            continue;
        }
        match &ev.logical_key {
            Key::Backspace => {
                input.text.pop();
            }
            Key::Space => input.text.push(' '),
            Key::Enter => input.focused = false,
            Key::Escape => input.focused = false,
            Key::Character(s) => {
                for ch in s.chars() {
                    if !ch.is_control() {
                        input.text.push(ch);
                    }
                }
            }
            _ => {}
        }
    }
}

fn update_path_text(
    input: Res<PathInput>,
    mut query: Query<(&mut Text, &mut TextColor), With<PathText>>,
) {
    if !input.is_changed() {
        return;
    }
    for (mut text, mut color) in &mut query {
        let display = if input.text.is_empty() {
            "<enter .mvsj path>".to_string()
        } else {
            input.text.clone()
        };
        let caret = if input.focused { "_" } else { "" };
        *text = Text::new(format!("{display}{caret}"));
        color.0 = if input.text.is_empty() {
            TEXT_DIM
        } else {
            TEXT_BRIGHT
        };
    }
}

// ---- Status bar ----------------------------------------------------------------

fn collect_errors(mut errors: MessageReader<MvsError>, mut status: ResMut<StatusBar>) {
    let collected: Vec<String> = errors.read().map(format_error).collect();
    if !collected.is_empty() {
        status.text = collected.join("  |  ");
        status.error = true;
    }
}

fn format_error(err: &MvsError) -> String {
    match err {
        MvsError::FileNotFound(p) => format!("File not found: {}", p.display()),
        MvsError::ParseError(s) => format!("Parse error: {s}"),
        MvsError::NoAtomsSelected {
            selector_description,
        } => format!("No atoms selected: {selector_description}"),
        MvsError::InvalidTransform => "Invalid transform matrix".to_string(),
        MvsError::UnsupportedNode { kind, reason } => {
            format!("Unsupported {kind:?}: {reason}")
        }
    }
}

fn render_status(
    status: Res<StatusBar>,
    mut query: Query<(&mut Text, &mut TextColor), With<StatusText>>,
) {
    if !status.is_changed() {
        return;
    }
    for (mut text, mut color) in &mut query {
        *text = Text::new(status.text.clone());
        color.0 = if status.error { STATUS_ERR } else { STATUS_OK };
    }
}

// ---- Tree inspector ------------------------------------------------------------

fn mark_tree_dirty_on_load(state_res: Res<MvsStateResource>, mut dirty: ResMut<TreeDirty>) {
    if state_res.is_changed() {
        dirty.0 = true;
    }
}

fn handle_tree_rows(
    query: Query<(&Interaction, &TreeRow), Changed<Interaction>>,
    mut collapse: ResMut<TreeCollapse>,
    mut dirty: ResMut<TreeDirty>,
) {
    for (interaction, row) in &query {
        if *interaction == Interaction::Pressed {
            if collapse.0.contains(&row.0) {
                collapse.0.remove(&row.0);
            } else {
                collapse.0.insert(row.0.clone());
            }
            dirty.0 = true;
        }
    }
}

fn rebuild_tree(
    mut dirty: ResMut<TreeDirty>,
    state_res: Res<MvsStateResource>,
    collapse: Res<TreeCollapse>,
    panel: Query<Entity, With<TreePanel>>,
    rows: Query<Entity, With<TreeRow>>,
    mut commands: Commands,
) {
    if !dirty.0 {
        return;
    }
    dirty.0 = false;

    let Ok(panel) = panel.single() else {
        return;
    };

    // Clear existing rows.
    for row in &rows {
        commands.entity(row).despawn();
    }

    let mut lines: Vec<(Vec<usize>, String, bool)> = Vec::new();
    if let Some(state) = &state_res.0 {
        collect_tree_lines(&state.root, &mut Vec::new(), 0, &collapse.0, &mut lines);
    } else {
        lines.push((vec![], "  (no state loaded)".to_string(), false));
    }

    commands.entity(panel).with_children(|p| {
        for (path, label, has_children) in lines {
            spawn_tree_row(p, path, label, has_children);
        }
    });
}

/// Depth-first flatten of the node tree into renderable lines, skipping the
/// children of collapsed nodes.
fn collect_tree_lines(
    node: &MvsNode,
    path: &mut Vec<usize>,
    depth: usize,
    collapsed: &std::collections::HashSet<Vec<usize>>,
    out: &mut Vec<(Vec<usize>, String, bool)>,
) {
    let children = node.children.as_deref().unwrap_or(&[]);
    let has_children = !children.is_empty();
    let is_collapsed = collapsed.contains(path);

    let chevron = if has_children {
        if is_collapsed {
            "▶"
        } else {
            "▼"
        }
    } else {
        "·"
    };
    let indent = "  ".repeat(depth);
    let label = format!(
        "{indent}{chevron} {:?}{}",
        node.kind,
        params_summary(&node.params)
    );
    out.push((path.clone(), label, has_children));

    if has_children && !is_collapsed {
        for (i, child) in children.iter().enumerate() {
            path.push(i);
            collect_tree_lines(child, path, depth + 1, collapsed, out);
            path.pop();
        }
    }
}

/// A short one-line summary of a node's params for the inspector.
fn params_summary(params: &Option<NodeParams>) -> String {
    let detail = match params {
        Some(NodeParams::DownloadParams(p)) => {
            let name = std::path::Path::new(&p.url)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.url.clone());
            Some(name)
        }
        Some(NodeParams::ParseParams(p)) => Some(format!("{:?}", p.format)),
        Some(NodeParams::StructureParams(p)) => Some(format!("{:?}", p.structure_type)),
        Some(NodeParams::ComponentInlineParams(p)) => Some(describe_selector(&p.selector)),
        Some(NodeParams::RepresentationParams(p)) => Some(format!("{:?}", p.representation_type)),
        Some(NodeParams::ColorInlineParams(p)) => Some(format!("{:?}", p.color)),
        Some(NodeParams::LabelInlineParams(p)) => Some(format!("\"{}\"", p.text)),
        _ => None,
    };
    match detail {
        Some(d) => format!("  [{d}]"),
        None => String::new(),
    }
}

fn describe_selector(selector: &ComponentSelector) -> String {
    match selector {
        ComponentSelector::Selector(s) => format!("{s:?}"),
        ComponentSelector::Expression(_) => "expr".to_string(),
        ComponentSelector::ExpressionList(v) => format!("{} exprs", v.len()),
    }
}

fn spawn_tree_row(
    parent: &mut ChildSpawnerCommands,
    path: Vec<usize>,
    label: String,
    has_children: bool,
) {
    let mut row = parent.spawn((
        TreeRow(path),
        Node {
            width: Val::Percent(100.0),
            padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
            ..default()
        },
        BackgroundColor(Color::NONE),
    ));
    // Only nodes with children are interactive (collapse toggle).
    if has_children {
        row.insert(Button);
    }
    row.with_children(|r| {
        r.spawn((
            Text::new(label),
            TextColor(TEXT_BRIGHT),
        ));
    });
}

// ---- UI layout -----------------------------------------------------------------

fn setup_ui(mut commands: Commands, presets: Res<Presets>) {
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            ..default()
        })
        .with_children(|root| {
            spawn_top_bar(root);
            spawn_preset_row(root, &presets);
            spawn_middle(root);
            spawn_status_bar(root);
        });
}

fn spawn_top_bar(root: &mut ChildSpawnerCommands) {
    root.spawn((
        Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
        ThemeBackgroundColor(tokens::PANE_HEADER_BG),
    ))
    .with_children(|bar| {
        bar.spawn((
            Text::new("MolViewSpec Viewer"),
            TextColor(TEXT_BRIGHT),
        ));

        // Editable file-path field (click to focus, then type).
        bar.spawn((
            Button,
            PathFieldButton,
            Node {
                min_width: Val::Px(360.0),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.12, 0.12, 0.14)),
            BorderColor::all(Color::srgb(0.35, 0.35, 0.40)),
        ))
        .with_children(|field| {
            field.spawn((
                Text::new("<enter .mvsj path>"),
                PathText,
                TextColor(TEXT_DIM),
            ));
        });

        spawn_text_button(bar, "Load", LoadButton);
    });
}

fn spawn_preset_row(root: &mut ChildSpawnerCommands, presets: &Presets) {
    root.spawn((
        Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(6.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
        ThemeBackgroundColor(tokens::PANE_BODY_BG),
    ))
    .with_children(|row| {
        row.spawn((
            Text::new("Presets:"),
            TextColor(TEXT_DIM),
            Node {
                width: Val::Px(70.0),
                ..default()
            },
        ));
        for (i, preset) in presets.0.iter().enumerate() {
            spawn_text_button(row, preset.name, PresetButton(i));
        }
    });
}

fn spawn_middle(root: &mut ChildSpawnerCommands) {
    // Middle row grows to fill; left sidebar over a transparent viewport.
    root.spawn(Node {
        width: Val::Percent(100.0),
        flex_grow: 1.0,
        flex_direction: FlexDirection::Row,
        ..default()
    })
    .with_children(|mid| {
        // Left sidebar: tree inspector.
        mid.spawn((
            Node {
                width: Val::Px(280.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            ThemeBackgroundColor(tokens::PANE_BODY_BG),
        ))
        .with_children(|side| {
            side.spawn((
                Text::new("MVS Tree"),
                TextColor(TEXT_DIM),
                Node {
                    margin: UiRect::bottom(Val::Px(6.0)),
                    ..default()
                },
            ));
            // Container whose children are the tree rows (rebuilt on each load).
            side.spawn((
                TreePanel,
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(1.0),
                    ..default()
                },
            ));
        });

        // Main viewport spacer (transparent; the 3D camera renders behind the UI).
        mid.spawn(Node {
            flex_grow: 1.0,
            height: Val::Percent(100.0),
            ..default()
        });
    });
}

fn spawn_status_bar(root: &mut ChildSpawnerCommands) {
    root.spawn((
        Node {
            width: Val::Percent(100.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
        ThemeBackgroundColor(tokens::PANE_HEADER_BG),
    ))
    .with_children(|bar| {
        bar.spawn((
            Text::new("Ready — pick a preset or enter a .mvsj path."),
            StatusText,
            TextColor(STATUS_OK),
        ));
    });
}

fn spawn_text_button<C: Component>(parent: &mut ChildSpawnerCommands, caption: &str, marker: C) {
    parent
        .spawn((
            Button,
            marker,
            Node {
                padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(BTN_NORMAL),
            BorderColor::all(Color::srgb(0.35, 0.35, 0.40)),
        ))
        .with_children(|btn| {
            btn.spawn((Text::new(caption), TextColor(TEXT_BRIGHT)));
        });
}

/// Hover/press feedback for the action buttons (not the tree rows or path field).
fn button_hover_colors(
    mut query: Query<
        (&Interaction, &mut BackgroundColor),
        (
            Changed<Interaction>,
            With<Button>,
            Without<TreeRow>,
            Without<PathFieldButton>,
        ),
    >,
) {
    for (interaction, mut bg) in &mut query {
        *bg = match *interaction {
            Interaction::Pressed => BackgroundColor(BTN_ACTIVE),
            Interaction::Hovered => BackgroundColor(BTN_HOVER),
            Interaction::None => BackgroundColor(BTN_NORMAL),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset must serialize to JSON that the executor's own parse path
    /// (`State::from_str`) accepts — this is the exact deserialization the
    /// `LoadMvsEvent::FromString` handler runs.
    fn assert_preset_parses(json: &str) -> State {
        State::from_str(json).expect("preset JSON must parse via State::from_str")
    }

    #[test]
    fn test_preset_basic_parses_and_has_focus() {
        let state = assert_preset_parses(&preset_basic("/tmp/1cbs.cif"));
        // A focus node lets the executor frame the structure.
        assert!(tree_has_camera_or_focus(&state.root));
    }

    #[test]
    fn test_preset_components_parses() {
        let state = assert_preset_parses(&preset_components("/tmp/1c0a.cif"));
        assert!(tree_has_camera_or_focus(&state.root));
        // protein + nucleic + ligand + 2 residues + focus = 6 components under the
        // single structure node.
        let mut lines = Vec::new();
        collect_tree_lines(
            &state.root,
            &mut Vec::new(),
            0,
            &std::collections::HashSet::new(),
            &mut lines,
        );
        let components = lines
            .iter()
            .filter(|(_, label, _)| label.contains("Component"))
            .count();
        assert_eq!(components, 6, "expected 6 components, got {components}");
    }

    #[test]
    fn test_preset_label_parses() {
        let state = assert_preset_parses(&preset_label("/tmp/1lap.cif"));
        let mut lines = Vec::new();
        collect_tree_lines(
            &state.root,
            &mut Vec::new(),
            0,
            &std::collections::HashSet::new(),
            &mut lines,
        );
        assert!(lines.iter().any(|(_, l, _)| l.contains("Label")));
    }

    #[test]
    fn test_preset_superposition_has_two_downloads_and_a_transform() {
        let state = assert_preset_parses(&preset_superposition("/tmp/4hhb.cif", "/tmp/1oj6.cif"));
        let mut lines = Vec::new();
        collect_tree_lines(
            &state.root,
            &mut Vec::new(),
            0,
            &std::collections::HashSet::new(),
            &mut lines,
        );
        let downloads = lines
            .iter()
            .filter(|(_, l, _)| l.contains("Download"))
            .count();
        assert_eq!(downloads, 2);
        assert!(lines.iter().any(|(_, l, _)| l.contains("Transform")));
    }

    #[test]
    fn test_preset_symmetry_parses() {
        // Builds a symmetry-typed structure; the executor degrades gracefully.
        let state = assert_preset_parses(&preset_symmetry("/tmp/1tqn.cif"));
        assert!(tree_has_camera_or_focus(&state.root));
    }

    #[test]
    fn test_collapsed_node_hides_descendants() {
        let state = assert_preset_parses(&preset_basic("/tmp/1cbs.cif"));
        // Collapse the root: only the root line should remain.
        let mut collapsed = std::collections::HashSet::new();
        collapsed.insert(Vec::<usize>::new());
        let mut lines = Vec::new();
        collect_tree_lines(&state.root, &mut Vec::new(), 0, &collapsed, &mut lines);
        assert_eq!(lines.len(), 1, "collapsing the root must hide all children");
    }
}
