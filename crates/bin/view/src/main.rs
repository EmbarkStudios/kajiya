mod opt;
mod persisted;
mod runtime;
mod scene;

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

    fn run(self) -> anyhow::Result<()> {
        let Self {
            mut persisted,
            mut runtime,
            kajiya,
        } = self;

        kajiya.run(|ctx| runtime.frame(ctx, &mut persisted))
    }
}

const APP_STATE_CONFIG_FILE_PATH: &str = "view_state.ron";

// If true, have an additional test/debug object in the scene, whose position can be changed by holding `Z` and moving the mouse.
const USE_TEST_DYNAMIC_OBJECT: bool = false;
const SPIN_TEST_DYNAMIC_OBJECT: bool = false;

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    let persisted: PersistedState = Default::default();

    let mut state = AppState::new(persisted, &opt)?;

    state.load_scene(&opt.scene)?;
    state.run()?;

    /*ron::ser::to_writer_pretty(
        File::create(APP_STATE_CONFIG_FILE_PATH)?,
        &state,
        Default::default(),
    )?;*/

    Ok(())
}
