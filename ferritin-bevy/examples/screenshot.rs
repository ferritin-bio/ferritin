use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Capturing, Screenshot},
    window::SystemCursorIcon,
    winit::cursor::CursorIcon,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_screenshot_and_exit))
        .run();
}

fn handle_screenshot_and_exit(world: &mut World) {
    let frame = world.resource::<Time>().elapsed();

    // Take screenshot on frame 10
    if frame.as_secs() > 1 {
        info!("Taking screenshot at frame {}", frame.as_secs());
        let path = format!("./screenshot-{}.png", frame.as_secs());
        world
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    // Check if screenshot is done and exit
    let screenshot_count = world
        .query_filtered::<Entity, With<Capturing>>()
        .iter(world)
        .count();

    if frame.as_secs() > 1 && screenshot_count == 0 {
        info!("Screenshot complete, exiting");
        world.send_event(AppExit::Success);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(5.0, 5.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
    ));
    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
