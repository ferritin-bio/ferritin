use anyhow::Result;
use bevy::prelude::*;
use ferritin_bevy::{Structure, ToRerun};
use ferritin_core::load_structure;
use ferritin_test_data::TestFile;
use rerun::demo_util::color_spiral;
use rerun::{self as rr};
use std::f32::consts::TAU;

fn main() -> Result<()> {
    let (molfile, _handle) = TestFile::protein_01().create_temp()?;

    let _chalky = StandardMaterial {
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

    let ac = load_structure(molfile).unwrap();
    let structure = Structure::builder().pdb(ac).build();
    let mesh: Mesh = structure.to_mesh();

    // mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    // Convert Bevy Mesh to Rerun Mesh3D
    let rerun_mesh = mesh.to_rerun();

    println!("Converted Bevy mesh to Rerun mesh3d: {:?}", rerun_mesh);

    let (rec, storage) = rerun::RecordingStreamBuilder::new("rerun_example_dna_abacus").memory()?;
    let main_thread_token = rerun::MainThreadToken::i_promise_i_am_on_the_main_thread();
    //re_log::setup_logging();

    const NUM_POINTS: usize = 100;

    let (points1, colors1) = color_spiral(NUM_POINTS, 2.0, 0.02, 0.0, 0.1);
    let (points2, colors2) = color_spiral(NUM_POINTS, 2.0, 0.02, TAU * 0.5, 0.1);

    rec.log(
        "dna/structure/left",
        &rerun::Points3D::new(points1.iter().copied())
            .with_colors(colors1)
            .with_radii([0.08]),
    )?;
    rec.log(
        "dna/structure/right",
        &rerun::Points3D::new(points2.iter().copied())
            .with_colors(colors2)
            .with_radii([0.08]),
    )?;

    rerun::native_viewer::show(main_thread_token, storage.take())?;

    Ok(())
}
