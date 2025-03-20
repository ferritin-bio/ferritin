use anyhow::Result;
use bevy::prelude::*;
use ferritin_bevy::{
    ColorScheme, RenderOptions, Structure, StructurePlugin, StructureSettings, ToRerun,
};
use ferritin_core::load_structure;
use ferritin_test_data::TestFile;
use rerun::demo_util::color_spiral;
use rerun::{self as rr, Mesh3D};
use std::f32::consts::TAU;

fn main() -> Result<()> {
    // start Rerun
    let (rec, storage) = rerun::RecordingStreamBuilder::new("rerun_protein_rendering").memory()?;
    let main_thread_token = rerun::MainThreadToken::i_promise_i_am_on_the_main_thread();

    // Load the structure
    let (molfile, _handle) = TestFile::protein_01().create_temp()?;
    let ac = load_structure(molfile).unwrap();

    // Define a few materials
    let chalky = StandardMaterial {
        base_color: Color::srgb(0.4, 0.4, 0.4), // Light gray color
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

    // Add one strucutre
    let structure = Structure::builder()
        .pdb(ac.clone())
        .material(chalky.clone())
        .rendertype(RenderOptions::BallAndStick)
        .color_scheme(ColorScheme::ByAtomType)
        .build();

    let mesh: Mesh = structure.to_mesh();
    let rerun_mesh: Mesh3D = mesh.to_rerun().unwrap();
    rec.log("protein/structure/ball_and_stick", &rerun_mesh)?;

    // Add another
    let structure = Structure::builder()
        .pdb(ac.clone())
        .material(chalky.clone())
        .rendertype(RenderOptions::Solid)
        .color_scheme(ColorScheme::ByAtomType)
        .build();

    let mesh: Mesh = structure.to_mesh();
    let rerun_mesh: Mesh3D = mesh.to_rerun().unwrap();
    rec.log("protein/structure/solic", &rerun_mesh)?;

    // gram the main thread
    rerun::native_viewer::show(main_thread_token, storage.take())?;

    Ok(())
}
