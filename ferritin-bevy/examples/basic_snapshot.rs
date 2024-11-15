//!  Example allowing custom colors and rendering options
use bevy::{
    app::AppExit,
    prelude::*,
    render::view::screenshot::{save_to_disk, Capturing, Screenshot},
    window::SystemCursorIcon,
    winit::cursor::CursorIcon,
};
use ferritin_bevy::{ColorScheme, RenderOptions, StructurePlugin, StructureSettings};

fn main() {
    let chalky = StandardMaterial {
        base_color: Color::srgb(0.9, 0.9, 0.9), // Light gray color
        perceptual_roughness: 1.0,              // Maximum roughness for a matte look
        metallic: 0.0,                          // No metallic properties
        reflectance: 0.1,                       // Low reflectance
        specular_transmission: 0.0,             // No specular transmission
        thickness: 0.0,                         // No thickness (for transparency)
        ior: 1.5,                               // Index of refraction (standard for most materials)
        alpha_mode: AlphaMode::Opaque,          // Fully opaque
        cull_mode: None,                        // Don't cull any faces
        ..default()                             // Use defaults for other properties
    };

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(StructurePlugin::new().with_file(
            "examples/1fap.cif",
            Some(StructureSettings {
                render_type: RenderOptions::BallAndStick,
                color_scheme: ColorScheme::ByAtomType,
                material: chalky,
            }),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (take_screenshot_and_exit, screenshot_saving))
        .run();
}

fn screenshot_saving(
    mut commands: Commands,
    screenshot_saving: Query<Entity, With<Capturing>>,
    windows: Query<Entity, With<Window>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    match screenshot_saving.iter().count() {
        0 => {
            commands.entity(window).remove::<CursorIcon>();
        }
        x if x > 0 => {
            commands
                .entity(window)
                .insert(CursorIcon::from(SystemCursorIcon::Progress));
        }
        _ => {}
    }
}

fn take_screenshot_and_exit(mut commands: Commands, mut exit: EventWriter<AppExit>) {
    // Wait a few frames to make sure everything is properly initialized
    static mut FRAME_COUNT: u32 = 0;
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT < 3 {
            // Wait for 3 frames
            return;
        }
    }

    let path = format!("./screenshot-{}.png", "01");
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));

    exit.send(AppExit::Success);
}

#[derive(Component)]
struct MainCamera;

fn setup(mut commands: Commands) {
    // Add a camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 50.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

    // Key Light
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.9, 0.9),
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
    ));

    // Fill Light
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.8, 0.8, 1.0),
            illuminance: 5000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.5, -0.5, 0.0)),
    ));

    // Backlight
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(0.9, 0.9, 1.0),
            illuminance: 3000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            0.0,
            std::f32::consts::PI,
            0.0,
        )),
    ));

    // Add a light
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Spotlight
    commands.spawn((
        SpotLight {
            intensity: 10000.0,
            color: Color::srgb(0.8, 1.0, 0.8),
            shadows_enabled: true,
            outer_angle: 0.6,
            ..default()
        },
        Transform::from_xyz(-4.0, 5.0, -4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
