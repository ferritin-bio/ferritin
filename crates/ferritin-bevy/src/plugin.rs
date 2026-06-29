//! Module for loading PDBs into Bevy via the Plugin system
//!
//! Over time this would be a good candidate for factoring out
//!
use super::{ColorScheme, RenderOptions, Structure};
use bevy::prelude::*;
use ferritin_core::load_model;
use std::path::Path;
use std::path::PathBuf;

#[derive(Clone)]
pub struct StructureSettings {
    pub render_type: RenderOptions,
    pub color_scheme: ColorScheme,
    pub material: StandardMaterial,
}
impl Default for StructureSettings {
    fn default() -> Self {
        Self {
            render_type: RenderOptions::Solid,
            color_scheme: ColorScheme::Solid(Color::WHITE),
            material: StandardMaterial::default(),
        }
    }
}

// adding this for integration with Bevy
pub struct StructurePlugin {
    initial_files: Vec<(PathBuf, StructureSettings)>,
}

impl StructurePlugin {
    pub fn new() -> Self {
        Self {
            initial_files: Vec::new(),
        }
    }
    pub fn with_file<P: Into<PathBuf>>(
        mut self,
        path: P,
        settings: Option<StructureSettings>,
    ) -> Self {
        self.initial_files
            .push((path.into(), settings.unwrap_or_default()));
        self
    }
}

// adding this for integration with Bevy
impl Plugin for StructurePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(StructureFiles(self.initial_files.clone()))
            .add_systems(Startup, load_initial_proteins)
            .add_message::<LoadProteinEvent>();
    }
}

#[derive(Resource)]
struct StructureFiles(Vec<(PathBuf, StructureSettings)>);

#[derive(Message)]
pub struct LoadProteinEvent(pub PathBuf);

fn load_initial_proteins(
    structure_files: Res<StructureFiles>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (file_path, settings) in &structure_files.0 {
        // check valid filepath
        if !Path::new(file_path).exists() {
            eprintln!("Error: File not found: {:?}", file_path);
            continue;
        }

        let model = load_model(file_path).unwrap();

        let structure = Structure::builder()
            .pdb(model)
            .rendertype(settings.render_type.clone())
            .color_scheme(settings.color_scheme.clone())
            .material(settings.material.clone())
            .build();

        let mesh = structure.to_mesh();
        let material = structure.get_material();

        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(material)),
        ));
    }
}
