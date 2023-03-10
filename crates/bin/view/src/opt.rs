use std::path::PathBuf;

use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "view", about = "Kajiya scene viewer.")]
pub struct Opt {
    #[structopt(long, default_value = "1920")]
    pub width: u32,

    #[structopt(long, default_value = "1080")]
    pub height: u32,

    #[structopt(long, default_value = "1.0")]
    pub temporal_upsampling: f32,

    #[structopt(long)]
    pub scene: Option<PathBuf>,

    #[structopt(long)]
    pub mesh: Option<PathBuf>,

    #[structopt(long, default_value = "1.0")]
    pub mesh_scale: f32,

    #[structopt(long)]
    pub no_vsync: bool,

    #[structopt(long)]
    pub no_window_decorations: bool,

    #[structopt(long)]
    pub fullscreen: bool,

    #[structopt(long)]
    pub graphics_debugging: bool,

    #[structopt(long)]
    pub physical_device_index: Option<usize>,

    #[structopt(long)]
    pub keymap: Option<PathBuf>,
}
