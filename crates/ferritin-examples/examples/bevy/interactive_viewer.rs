//! Interactive protein viewer with representation and color controls.
//!
//! Uses bevy_feathers for dark-theme panel styling and standard Bevy UI buttons
//! for switching representation type and color scheme at runtime.
//!
//! Run with:
//!   cargo run --example bevy_interactive_viewer --features bevy -p ferritin-examples

use anyhow::Result;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy_feathers::{
    dark_theme::create_dark_theme,
    theme::{ThemeBackgroundColor, UiTheme},
    tokens, FeathersPlugins,
};
use ferritin_core::{load_structure, AtomCollection};
use ferritin_structure_mesh::{ColorScheme, RenderOptions, Structure};
use ferritin_test_data::TestFile;

// ---- Resources -----------------------------------------------------------------

#[derive(Resource)]
struct ViewerState {
    render_type: RenderOptions,
    color_scheme: ColorScheme,
}

#[derive(Resource)]
struct ProteinData(AtomCollection);

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

// ---- Colour constants for manual button styling --------------------------------

const BTN_NORMAL: Color = Color::srgb(0.18, 0.18, 0.20);
const BTN_HOVER: Color = Color::srgb(0.28, 0.28, 0.32);
const BTN_ACTIVE: Color = Color::srgb(0.22, 0.42, 0.64);

// ---- Entry point ---------------------------------------------------------------

fn main() -> Result<()> {
    let (molfile, _handle) = TestFile::protein_01().create_temp()?;
    let mut ac = load_structure(&molfile)?;
    ac.connect_via_residue_names();

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FeathersPlugins)
        // Activate the feathers dark theme.
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(ViewerState {
            render_type: RenderOptions::Solid,
            color_scheme: ColorScheme::ByAtomType,
        })
        .insert_resource(ProteinData(ac))
        .add_systems(Startup, (setup_scene, setup_ui))
        // rebuild_protein runs whenever ViewerState changes (including initial insertion).
        .add_systems(
            Update,
            (
                rebuild_protein.run_if(resource_changed::<ViewerState>),
                handle_button_interactions,
            ),
        )
        .run();

    Ok(())
}

// ---- 3-D scene -----------------------------------------------------------------

fn setup_scene(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 50.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
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

    commands.spawn((
        Mesh3d(meshes.add(structure.to_mesh())),
        MeshMaterial3d(materials.add(structure.get_material())),
        ProteinMesh,
    ));
}

/// Update ViewerState when a control button is pressed; update hover colours too.
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
                        ("Wireframe", ButtonAction::SetRender(RenderOptions::Wireframe)),
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
                            ButtonAction::SetColor(ColorScheme::Solid(Color::srgb(
                                0.0, 0.8, 0.8,
                            ))),
                        ),
                        (
                            "Gold",
                            ButtonAction::SetColor(ColorScheme::Solid(Color::srgb(
                                1.0, 0.8, 0.2,
                            ))),
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
