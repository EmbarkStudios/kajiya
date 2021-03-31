use glam::Quat;
use kajiya_asset::mesh::{pack_triangle_mesh, GpuImage, LoadGltfScene, PackedTriMesh};
use std::{collections::HashSet, fs::File, path::PathBuf};

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

    #[structopt(short = "o")]
    output_name: String,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let lazy_cache = LazyCache::create();

    std::fs::create_dir_all("baked")?;

    {
        let mesh = LoadGltfScene {
            path: opt.scene,
            scale: opt.scale,
            //rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            rotation: Quat::identity(),
        }
        .into_lazy();

        let mesh: PackedTriMesh::Proto =
            pack_triangle_mesh(&*smol::block_on(mesh.eval(&lazy_cache))?);

        mesh.flatten_into(&mut File::create(format!(
            "baked/{}.mesh",
            opt.output_name
        ))?);
        let unique_images: Vec<Lazy<GpuImage::Proto>> = mesh
            .maps
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let lazy_cache = &lazy_cache;
        let images = unique_images.iter().cloned().map(|img| async move {
            let loaded = smol::spawn(img.eval(lazy_cache)).await?;

            loaded.flatten_into(&mut File::create(format!(
                "baked/{:8.8x}.image",
                img.identity()
            ))?);

            anyhow::Result::<()>::Ok(())
        });

        smol::block_on(futures::future::try_join_all(images)).expect("Failed to load mesh images");
    }

    Ok(())
}
