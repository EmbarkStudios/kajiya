mod gui;
mod keymap;
mod misc;
mod opt;
mod persisted;
mod runtime;
mod scene;
mod sequence;

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use kajiya_simple::*;
use opt::*;
use persisted::*;
use runtime::*;

use structopt::StructOpt;

struct AppState {
    persisted: PersistedState,
    runtime: RuntimeState,
    kajiya: SimpleMainLoop,
}

impl AppState {
    fn new(mut persisted: PersistedState, opt: &Opt) -> anyhow::Result<Self> {
        let mut kajiya = SimpleMainLoop::builder()
            .resolution([opt.width, opt.height])
            .vsync(!opt.no_vsync)
            .graphics_debugging(opt.graphics_debugging)
            .physical_device_index(opt.physical_device_index)
            .temporal_upsampling(opt.temporal_upsampling)
            .default_log_level(log::LevelFilter::Info)
            .fullscreen(opt.fullscreen.then_some(FullscreenMode::Exclusive))
            .build(
                WindowBuilder::new()
                    .with_title("kajiya")
                    .with_resizable(false)
                    .with_decorations(!opt.no_window_decorations),
            )?;

        let runtime = RuntimeState::new(&mut persisted, &mut kajiya.world_renderer, opt);

        Ok(Self {
            persisted,
            runtime,
            kajiya,
        })
    }

    fn load_scene(&mut self, scene_path: &Path) -> anyhow::Result<()> {
        self.runtime.load_scene(
            &mut self.persisted,
            &mut self.kajiya.world_renderer,
            scene_path,
        )
    }

    fn add_standalone_mesh(&mut self, path: PathBuf, mesh_scale: f32) -> anyhow::Result<()> {
        self.runtime.add_mesh_instance(
            &mut self.persisted,
            &mut self.kajiya.world_renderer,
            MeshSource::File(path),
            SceneElementTransform {
                position: Vec3::ZERO,
                rotation_euler_degrees: Vec3::ZERO,
                scale: Vec3::splat(mesh_scale),
            },
        )
    }

    fn run(self) -> anyhow::Result<PersistedState> {
        let Self {
            mut persisted,
            mut runtime,
            kajiya,
        } = self;

        kajiya.run(|ctx| runtime.frame(ctx, &mut persisted))?;

        Ok(persisted)
    }
}

const APP_STATE_CONFIG_FILE_PATH: &str = "view_state.ron";

fn main() -> anyhow::Result<()> {
    set_vfs_mount_point("/meshes", "assets/meshes");

    let opt = Opt::from_args();

    let mut persisted: PersistedState = File::open(APP_STATE_CONFIG_FILE_PATH)
        .map_err(|err| anyhow::anyhow!(err))
        .and_then(|file| Ok(ron::de::from_reader(file)?))
        .unwrap_or_default();

    // If supplying a new scene, clear the previous one.
    if opt.scene.is_some() || opt.mesh.is_some() {
        persisted.scene = SceneState::default();
    }

    let mut state = AppState::new(persisted, &opt)?;

    if let Some(scene) = opt.scene.as_ref() {
        state.load_scene(scene)?;
    } else if let Some(mesh) = opt.mesh.as_ref() {
        state.add_standalone_mesh(mesh.clone(), opt.mesh_scale)?;
    }

    let state = state.run()?;

    ron::ser::to_writer_pretty(
        File::create(APP_STATE_CONFIG_FILE_PATH)?,
        &state,
        Default::default(),
    )?;

    Ok(())
}
