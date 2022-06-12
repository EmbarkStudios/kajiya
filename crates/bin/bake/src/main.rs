use anyhow::Result;
use kajiya_asset_pipe::*;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "bake", about = "Kanelbullar")]
struct Opt {
    #[structopt(long, parse(from_os_str))]
    scene: PathBuf,

    #[structopt(long, default_value = "1.0")]
    scale: f32,

    #[structopt(short = "o")]
    output_name: String,
}

fn main() -> Result<()> {
    env_logger::init();

    let opt = Opt::from_args();

    process_mesh_asset(MeshAssetProcessParams {
        path: opt.scene,
        output_name: opt.output_name,
        scale: opt.scale,
    })
}
