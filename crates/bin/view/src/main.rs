mod gui;
mod misc;
mod opt;
mod persisted;
mod runtime;
mod scene;
mod sequence;

use std::fs::File;

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
    fn new(persisted: PersistedState, opt: &Opt) -> anyhow::Result<Self> {
        let kajiya = SimpleMainLoop::builder()
            .resolution([opt.width, opt.height])
            .vsync(!opt.no_vsync)
            .graphics_debugging(!opt.no_debug)
            .physical_device_index(opt.physical_device_index)
            .temporal_upsampling(opt.temporal_upsampling)
            .default_log_level(log::LevelFilter::Info)
            .fullscreen(opt.fullscreen.then(|| FullscreenMode::Exclusive))
            .build(
                WindowBuilder::new()
                    .with_title("kajiya")
                    .with_resizable(false)
                    .with_decorations(!opt.no_window_decorations),
            )?;

        let runtime = RuntimeState::new(&persisted, opt);

        Ok(Self {
            persisted,
            runtime,
            kajiya,
        })
    }

    fn load_scene(&mut self, scene_name: &str) -> anyhow::Result<()> {
        self.runtime
            .load_scene(&mut self.kajiya.world_renderer, scene_name)
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
    let opt = Opt::from_args();

    let persisted: PersistedState = File::open(APP_STATE_CONFIG_FILE_PATH)
        .map_err(|err| anyhow::anyhow!(err))
        .and_then(|file| Ok(ron::de::from_reader(file)?))
        .unwrap_or_default();

    let mut state = AppState::new(persisted, &opt)?;

    state.load_scene(&opt.scene)?;
    let state = state.run()?;

    ron::ser::to_writer_pretty(
        File::create(APP_STATE_CONFIG_FILE_PATH)?,
        &state,
        Default::default(),
    )?;

    Ok(())
}
