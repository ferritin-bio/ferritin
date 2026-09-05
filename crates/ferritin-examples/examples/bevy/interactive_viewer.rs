//! Interactive protein viewer with representation and color controls.
//!
//! Uses bevy_feathers for dark-theme panel styling and standard Bevy UI buttons
//! for switching representation type and color scheme at runtime.
//!
//! Controls:
//!   Left-drag   — orbit (rotate around molecule)
//!   Right-drag  — pan
//!   Scroll      — zoom
//!
//! Run with:
//!   cargo run --example bevy_interactive_viewer --features bevy -p ferritin-examples

use anyhow::Result;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy_feathers::{
    FeathersPlugins,
    dark_theme::create_dark_theme,
    theme::{ThemeBackgroundColor, UiTheme},
    tokens,
};
use ferritin_bevy::{ColorScheme, RenderOptions, Structure};
use ferritin_core::{Model, load_model};
use ferritin_test_data::TestFile;

// ---- Resources -----------------------------------------------------------------

#[derive(Resource)]
struct ViewerState {
    render_type: RenderOptions,
    color_scheme: ColorScheme,
}

#[derive(Resource)]
struct ProteinData(Model);

// ---- Components ----------------------------------------------------------------

/// Marker for the spawned protein mesh entity so we can despawn/respawn it.
#[derive(Component)]
struct ProteinMesh;

/// Payload attached to each control button.
#[derive(Component, Clone)]
enum ButtonAction {
    SetRender(RenderOptions),
    SetColor(ColorScheme),
}

/// Orbit camera state. The camera always looks at `focus` from a distance of
/// `radius`, oriented by `yaw` (around Y) and `pitch` (elevation).
#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        // Initial camera position equivalent to (0, 50, 100) looking at ZERO.
        let radius = 50.0_f32.hypot(100.0);
        Self {
            focus: Vec3::ZERO,
            radius,
            yaw: 0.0,
            pitch: (50.0 / radius).asin(),
        }
    }
}

// ---- Colour constants for manual button styling --------------------------------

const BTN_NORMAL: Color = Color::srgb(0.18, 0.18, 0.20);
const BTN_HOVER: Color = Color::srgb(0.28, 0.28, 0.32);
const BTN_ACTIVE: Color = Color::srgb(0.22, 0.42, 0.64);

// ---- Entry point ---------------------------------------------------------------

fn main() -> Result<()> {
    let (molfile, _handle) = TestFile::protein_01().create_temp()?;
    // load_model already calls connect_via_residue_names internally.
    let model = load_model(&molfile)?;

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FeathersPlugins)
        // Activate the feathers dark theme.
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(ViewerState {
            render_type: RenderOptions::Solid,
            color_scheme: ColorScheme::ByAtomType,
        })
        .insert_resource(ProteinData(model))
        // rebuild_protein runs whenever ViewerState changes (including initial insertion).
        .add_systems(Startup, (setup_scene, setup_ui))
        .add_systems(
            Update,
            (
                rebuild_protein.run_if(resource_changed::<ViewerState>),
                handle_button_interactions,
                orbit_camera,
            ),
        )
        .run();

    Ok(())
}

// ---- 3-D scene -----------------------------------------------------------------

fn setup_scene(mut commands: Commands) {
    // Compute initial camera transform from OrbitCamera defaults.
    let orbit = OrbitCamera::default();
    let cam_pos = orbit_position(&orbit);

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam_pos).looking_at(orbit.focus, Vec3::Y),
        orbit,
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
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            0.0,
            std::f32::consts::PI,
            0.0,
        )),
    ));
}

/// Despawn any existing protein mesh and spawn a new one reflecting the current ViewerState.
/// Also fires on the first Update frame (resource is "changed" when first inserted).
fn rebuild_protein(
    protein: Res<ProteinData>,
    state: Res<ViewerState>,
    existing: Query<Entity, With<ProteinMesh>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }

    let structure = Structure::builder()
        .pdb(protein.0.clone())
        .rendertype(state.render_type.clone())
        .color_scheme(state.color_scheme.clone())
        .build();

    // Center the molecule on the origin so the orbit camera always looks at it.
    let centroid = structure.centroid();

    commands.spawn((
        Mesh3d(meshes.add(structure.to_mesh())),
        MeshMaterial3d(materials.add(structure.get_material())),
        Transform::from_translation(-centroid),
        ProteinMesh,
    ));
}

/// Update ViewerState when a control button is pressed; update hover colours too.
#[allow(clippy::type_complexity)] // idiomatic Bevy query type
fn handle_button_interactions(
    mut query: Query<
        (&Interaction, &ButtonAction, &mut BackgroundColor),
        (Changed<Interaction>, With<Button>),
    >,
    mut state: ResMut<ViewerState>,
) {
    for (interaction, action, mut bg) in &mut query {
        match *interaction {
            Interaction::Pressed => {
                *bg = BackgroundColor(BTN_ACTIVE);
                match action {
                    ButtonAction::SetRender(rt) => state.render_type = rt.clone(),
                    ButtonAction::SetColor(cs) => state.color_scheme = cs.clone(),
                }
            }
            Interaction::Hovered => *bg = BackgroundColor(BTN_HOVER),
            Interaction::None => *bg = BackgroundColor(BTN_NORMAL),
        }
    }
}

/// Pan / orbit / zoom the camera via mouse input.
///
/// Left-drag  → orbit   Right-drag → pan   Scroll → zoom
fn orbit_camera(
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
) {
    let delta = motion.delta;

    let scroll_delta = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y * 5.0,
        MouseScrollUnit::Pixel => scroll.delta.y * 0.1,
    };

    for (mut transform, mut orbit) in &mut query {
        if mouse_input.pressed(MouseButton::Left) && delta != Vec2::ZERO {
            orbit.yaw -= delta.x * 0.005;
            orbit.pitch = (orbit.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
        }

        if mouse_input.pressed(MouseButton::Right) && delta != Vec2::ZERO {
            let right = transform.right();
            let up = transform.up();
            let pan_scale = orbit.radius * 0.001;
            orbit.focus += (right * -delta.x + up * delta.y) * pan_scale;
        }

        if scroll_delta != 0.0 {
            orbit.radius = (orbit.radius - scroll_delta).max(1.0);
        }

        let pos = orbit_position(&orbit);
        transform.translation = pos;
        transform.look_at(orbit.focus, Vec3::Y);
    }
}

/// Compute world-space camera position from orbit parameters.
fn orbit_position(orbit: &OrbitCamera) -> Vec3 {
    let x = orbit.radius * orbit.yaw.sin() * orbit.pitch.cos();
    let y = orbit.radius * orbit.pitch.sin();
    let z = orbit.radius * orbit.yaw.cos() * orbit.pitch.cos();
    orbit.focus + Vec3::new(x, y, z)
}

// ---- UI panel ------------------------------------------------------------------

fn setup_ui(mut commands: Commands) {
    // Full-screen root; flex column, panel pinned to bottom.
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::FlexEnd,
            ..default()
        })
        .with_children(|root| {
            // Control panel – feathers PANE_BODY_BG token sets background via theme observer.
            root.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(10.0)),
                    row_gap: Val::Px(8.0),
                    ..default()
                },
                BackgroundColor(Color::NONE), // overwritten by feathers ThemeBackgroundColor
                ThemeBackgroundColor(tokens::PANE_BODY_BG),
            ))
            .with_children(|panel| {
                spawn_control_row(
                    panel,
                    "Representation",
                    &[
                        (
                            "Wireframe",
                            ButtonAction::SetRender(RenderOptions::Wireframe),
                        ),
                        ("Cartoon", ButtonAction::SetRender(RenderOptions::Cartoon)),
                        (
                            "Ball+Stick",
                            ButtonAction::SetRender(RenderOptions::BallAndStick),
                        ),
                        ("Solid", ButtonAction::SetRender(RenderOptions::Solid)),
                        ("Putty", ButtonAction::SetRender(RenderOptions::Putty)),
                    ],
                );
                spawn_control_row(
                    panel,
                    "Color",
                    &[
                        ("By Atom", ButtonAction::SetColor(ColorScheme::ByAtomType)),
                        (
                            "White",
                            ButtonAction::SetColor(ColorScheme::Solid(Color::WHITE)),
                        ),
                        (
                            "Teal",
                            ButtonAction::SetColor(ColorScheme::Solid(Color::srgb(0.0, 0.8, 0.8))),
                        ),
                        (
                            "Gold",
                            ButtonAction::SetColor(ColorScheme::Solid(Color::srgb(1.0, 0.8, 0.2))),
                        ),
                    ],
                );
            });
        });
}

fn spawn_control_row(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    buttons: &[(&str, ButtonAction)],
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(6.0),
            ..default()
        })
        .with_children(|row| {
            // Section label
            row.spawn((
                Text::new(format!("{label}:")),
                TextColor(Color::srgb(0.75, 0.75, 0.75)),
                Node {
                    width: Val::Px(120.0),
                    ..default()
                },
            ));

            for (caption, action) in buttons {
                row.spawn((
                    Button,
                    action.clone(),
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
                    btn.spawn((
                        Text::new(*caption),
                        TextColor(Color::srgb(0.90, 0.90, 0.90)),
                    ));
                });
            }
        });
}
