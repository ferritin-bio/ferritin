//!  Example allowing custom colors and rendering options
use bevy::app::AppExit;
use bevy::prelude::*;
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
        .add_systems(Update, take_screenshot_and_exit)
        .run();
}

fn take_screenshot_and_exit(
    mut exit: EventWriter<AppExit>,
    windows: Query<&Window>,
    images: Res<Assets<Image>>,
    camera: Query<(&Camera, &GlobalTransform)>,
) {
    // Wait a few frames to make sure everything is properly initialized
    static mut FRAME_COUNT: u32 = 0;
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT < 3 {
            // Wait for 3 frames
            return;
        }
    }
    save_screenshot(windows, images, camera);
    println!("Screenshot taken!");
    exit.send(AppExit::Success);
}

fn save_screenshot(
    windows: Query<&Window>,
    images: Res<Assets<Image>>,
    camera: Query<(&Camera, &GlobalTransform)>,
) {
    let window = windows.single();
    let (camera, camera_transform) = camera.single();

    if let Some(image) = camera.texture.as_ref() {
        if let Some(image) = images.get(image) {
            let image_buffer = image::RgbaImage::from_raw(
                window.width() as u32,
                window.height() as u32,
                image.data.clone(),
            )
            .unwrap();

            image_buffer.save("screenshot.png").unwrap();
        }
    }
}

#[derive(Component)]
struct MainCamera;

fn setup(mut commands: Commands) {
    // Add a camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 50.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        MainCamera,
    ));

    // Key Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::srgb(1.0, 0.9, 0.9),
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
        ..default()
    });

    // Fill Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::srgb(0.8, 0.8, 1.0),
            illuminance: 5000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.5, -0.5, 0.0)),
        ..default()
    });

    // Back Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::srgb(0.9, 0.9, 1.0),
            illuminance: 3000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            0.0,
            std::f32::consts::PI,
            0.0,
        )),
        ..default()
    });

    // Add a light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // Spot light
    commands.spawn(SpotLightBundle {
        spot_light: SpotLight {
            intensity: 10000.0,
            color: Color::srgb(0.8, 1.0, 0.8),
            shadows_enabled: true,
            outer_angle: 0.6,
            ..default()
        },
        transform: Transform::from_xyz(-4.0, 5.0, -4.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
}
