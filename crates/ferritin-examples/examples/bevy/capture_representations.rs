//! Headless-ish screenshot harness for ferritin-ala.
//!
//! Renders every `RepresentationTypeT` x `ColorThemeT` combination on 4hhb
//! (hemoglobin: 4 chains, mixed secondary structure), plus the 5
//! `bevy_molviewspec_viewer` presets, and saves one PNG per scene to
//! `docs/screenshots/ferritin-ala/`. Uses Bevy's built-in GPU screenshot
//! (render-to-texture readback), which needs no OS screen-recording
//! permission — only a real window/GPU context.
//!
//! Run with:
//!   cargo run --example bevy_capture_representations --features bevy -p ferritin-examples

use anyhow::Result;
use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use ferritin_bevy::{LoadMvsEvent, MvsPlugin, OrbitCamera};
use ferritin_molviewspec::molviewspec::nodes::{
    ColorNamesT, ColorT, ColorThemeT, ComponentExpression, ComponentSelector,
    ComponentSelectorT, ParseFormatT, ParseParams, RepresentationTypeT, State, StructureParams,
    StructureTypeT, TransformParams,
};
use ferritin_test_data::TestFile;
use tempfile::NamedTempFile;

const OUT_DIR: &str = "docs/screenshots/ferritin-ala";
/// Frames to let a freshly-loaded scene settle (mesh build + camera framing)
/// before the GPU readback fires.
const SETTLE_FRAMES: u32 = 15;

struct Shot {
    name: &'static str,
    json: String,
}

fn main() -> Result<()> {
    std::fs::create_dir_all(OUT_DIR)?;

    let mut temps: Vec<NamedTempFile> = Vec::new();
    let mut extract = |tf: TestFile| -> Result<String> {
        let (path, handle) = tf.create_temp()?;
        temps.push(handle);
        Ok(path)
    };

    let p_4hhb = extract(TestFile::mvs_4hhb())?;
    let p_1cbs = extract(TestFile::mvs_1cbs())?;
    let p_1c0a = extract(TestFile::mvs_1c0a())?;
    let p_1lap = extract(TestFile::mvs_1lap())?;
    let p_1oj6 = extract(TestFile::mvs_1oj6())?;
    let p_1tqn = extract(TestFile::mvs_1tqn())?;

    let mut shots = Vec::new();

    let reps = [
        RepresentationTypeT::BallAndStick,
        RepresentationTypeT::Cartoon,
        RepresentationTypeT::Line,
        RepresentationTypeT::Putty,
        RepresentationTypeT::Spacefill,
        RepresentationTypeT::Surface,
    ];
    let themes = [
        ColorThemeT::ElementSymbol,
        ColorThemeT::ChainId,
        ColorThemeT::SecondaryStructure,
        ColorThemeT::Uniform,
    ];
    for rep in &reps {
        for theme in &themes {
            let name: &'static str =
                Box::leak(format!("rep_{rep:?}_{theme:?}").into_boxed_str());
            shots.push(Shot {
                name,
                json: single_rep_state(&p_4hhb, rep.clone(), theme.clone()),
            });
        }
    }

    shots.push(Shot {
        name: "preset_basic",
        json: preset_basic(&p_1cbs),
    });
    shots.push(Shot {
        name: "preset_components",
        json: preset_components(&p_1c0a),
    });
    shots.push(Shot {
        name: "preset_label",
        json: preset_label(&p_1lap),
    });
    shots.push(Shot {
        name: "preset_superposition",
        json: preset_superposition(&p_4hhb, &p_1oj6),
    });
    shots.push(Shot {
        name: "preset_symmetry",
        json: preset_symmetry(&p_1tqn),
    });

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MvsPlugin)
        .insert_resource(ShotQueue {
            shots: shots.into(),
            state: QueueState::Idle,
        })
        .insert_resource(PresetTempFiles(temps))
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (sync_camera, drive_queue, take_pending_screenshot).chain(),
        )
        .run();

    Ok(())
}

#[derive(Resource)]
#[allow(dead_code)]
struct PresetTempFiles(Vec<NamedTempFile>);

#[derive(PartialEq)]
enum QueueState {
    Idle,
    Settling(u32),
}

#[derive(Resource)]
struct ShotQueue {
    shots: std::collections::VecDeque<Shot>,
    state: QueueState,
}

fn drive_queue(
    mut queue: ResMut<ShotQueue>,
    mut loader: MessageWriter<LoadMvsEvent>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    match queue.state {
        QueueState::Idle => {
            let Some(shot) = queue.shots.pop_front() else {
                exit.write(AppExit::Success);
                return;
            };
            info!("loading {}", shot.name);
            loader.write(LoadMvsEvent::FromString(shot.json));
            commands.insert_resource(PendingName(shot.name));
            queue.state = QueueState::Settling(0);
        }
        QueueState::Settling(n) => {
            if n < SETTLE_FRAMES {
                queue.state = QueueState::Settling(n + 1);
            } else {
                queue.state = QueueState::Idle;
            }
        }
    }
}

#[derive(Resource)]
struct PendingName(&'static str);

fn take_pending_screenshot(
    mut commands: Commands,
    pending: Option<Res<PendingName>>,
    queue: Res<ShotQueue>,
) {
    // Fire the screenshot on the last settle frame only.
    if let QueueState::Settling(n) = queue.state {
        if n == SETTLE_FRAMES - 1 {
            if let Some(pending) = pending {
                let path = format!("{OUT_DIR}/{}.png", pending.0);
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            }
        }
    }
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

fn to_json(state: &State) -> String {
    serde_json::to_string(state).expect("MVS state serializes to JSON")
}

fn single_rep_state(path: &str, rep: RepresentationTypeT, theme: ColorThemeT) -> String {
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
        comp.representation_with_theme(rep, Some(theme));
        comp.focus(None, None);
    }
    to_json(&state)
}

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
            .color(ColorT::Named(ColorNamesT::Blue), sel(ComponentSelectorT::All));
        comp.focus(None, None);
    }
    to_json(&state)
}

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

        structure
            .component(sel(ComponentSelectorT::Protein))
            .unwrap()
            .representation(RepresentationTypeT::Cartoon)
            .unwrap()
            .color(orange, sel(ComponentSelectorT::All));
        structure
            .component(sel(ComponentSelectorT::Nucleic))
            .unwrap()
            .representation(RepresentationTypeT::Cartoon)
            .unwrap()
            .color(blue, sel(ComponentSelectorT::All));
        structure
            .component(sel(ComponentSelectorT::Ligand))
            .unwrap()
            .representation(RepresentationTypeT::BallAndStick)
            .unwrap()
            .color(green, sel(ComponentSelectorT::All));
        for seq in [217, 537] {
            let residue = ComponentExpression {
                auth_asym_id: Some("B".to_string()),
                auth_seq_id: Some(seq),
                ..Default::default()
            };
            let c = structure
                .component(ComponentSelector::Expression(residue))
                .unwrap();
            c.representation(RepresentationTypeT::BallAndStick)
                .unwrap()
                .color(red.clone(), sel(ComponentSelectorT::All));
            c.label("aaRS Class II Signature".to_string());
        }
        structure
            .component(sel(ComponentSelectorT::All))
            .unwrap()
            .focus(None, None);
    }
    to_json(&state)
}

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
        structure
            .component(sel(ComponentSelectorT::Protein))
            .unwrap()
            .representation(RepresentationTypeT::Cartoon)
            .unwrap()
            .color(ColorT::Named(ColorNamesT::Lightgray), sel(ComponentSelectorT::All));
        let c = structure.component(sel(ComponentSelectorT::Ion)).unwrap();
        c.representation(RepresentationTypeT::BallAndStick)
            .unwrap()
            .color(ColorT::Named(ColorNamesT::Magenta), sel(ComponentSelectorT::All));
        c.label("Catalytic metal site".to_string());
        c.focus(None, None);
    }
    to_json(&state)
}

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
        c.representation(RepresentationTypeT::Cartoon)
            .unwrap()
            .color(ColorT::Named(ColorNamesT::Steelblue), sel(ComponentSelectorT::All));
        c.focus(None, None);
    }
    {
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
        c.representation(RepresentationTypeT::Cartoon)
            .unwrap()
            .color(ColorT::Named(ColorNamesT::Orange), sel(ComponentSelectorT::All));
        // Focus the translated copy too, so the executor's focus union frames both
        // structures instead of leaving this one clipped off-frame (ferritin-t0h.6).
        c.focus(None, None);
    }
    to_json(&state)
}

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
        c.representation(RepresentationTypeT::Cartoon)
            .unwrap()
            .color(ColorT::Named(ColorNamesT::Seagreen), sel(ComponentSelectorT::All));
        c.focus(None, None);
    }
    to_json(&state)
}

// ---- Scene / camera --------------------------------------------------------

fn setup_scene(mut commands: Commands, orbit: Res<OrbitCamera>) {
    // Keep this lighting rig in sync with molviewspec_viewer's setup_scene so the
    // captured ground-truth reflects what the interactive app actually shows: an
    // ambient fill (per-camera in Bevy 0.19) plus a three-point directional rig
    // so surfaces facing away from the key lights don't read near-black
    // (ferritin-t0h.4).
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(orbit_position(&orbit)).looking_at(orbit.focus, orbit.up),
        AmbientLight {
            color: Color::WHITE,
            brightness: 600.0,
            ..default()
        },
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

fn orbit_position(orbit: &OrbitCamera) -> Vec3 {
    let x = orbit.radius * orbit.yaw.sin() * orbit.pitch.cos();
    let y = orbit.radius * orbit.pitch.sin();
    let z = orbit.radius * orbit.yaw.cos() * orbit.pitch.cos();
    orbit.focus + Vec3::new(x, y, z)
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

