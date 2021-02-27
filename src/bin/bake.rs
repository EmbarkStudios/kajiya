use std::{fs::File, io::Write, path::PathBuf};
use vicki::asset::mesh::{pack_triangle_mesh, FlatImage, LoadGltfScene};

use turbosloth::*;

use anyhow::Result;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "bake", about = "Kanelbullar")]
struct Opt {
    #[structopt(long, parse(from_os_str))]
    scene: PathBuf,

    #[structopt(long, default_value = "0.1")]
    scale: f32,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let lazy_cache = LazyCache::create();

    {
        let mesh = LoadGltfScene {
            path: opt.scene,
            scale: opt.scale,
        }
        .into_lazy();

        let mesh = smol::block_on(mesh.eval(&lazy_cache))?;
        let mut file = File::create("baked/derp.mesh")?;
        pack_triangle_mesh(&mesh).flatten_into(&mut file);
    }

    {
        let image = FlatImage::Proto {
            format: slingshot::ash::vk::Format::R32G32B32A32_SFLOAT,
            extent: [128, 128, 1],
            mips: vec![vec![1], vec![1, 2], vec![1, 2, 3]],
        };

        let mut file = File::create("baked/derp.image")?;
        image.flatten_into(&mut file);
    }

    Ok(())
}
