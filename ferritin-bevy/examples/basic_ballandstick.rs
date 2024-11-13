//!  Example allowing custom colors and rendering options
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

    let _metallic = StandardMaterial {
        base_color: Color::srgb(0.8, 0.8, 0.9), // Slight blue tint for a steel-like appearance
        metallic: 1.0,                          // Fully metallic
        perceptual_roughness: 0.1,              // Very smooth surface
        reflectance: 0.5,                       // Medium reflectance
        // emissive: Color::BLACK,                // No emission
        alpha_mode: AlphaMode::Opaque, // Fully opaque
        ior: 2.5,                      // Higher index of refraction for metals
        specular_transmission: 0.0,    // No light transmission
        thickness: 0.0,                // No thickness (for transparency)
        //cull_mode: Some(Face::Back),   // Cull back faces for better performance
        ..default() // Use defaults for other properties
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
        .run();
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

    // Backlight
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

    // Spotlight
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
