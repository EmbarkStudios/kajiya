use std::{collections::HashSet, fs::File, path::PathBuf, sync::Arc};
use vicki::asset::mesh::{pack_triangle_mesh, GpuImage, LoadGltfScene, PackedTriMesh};

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

        let mesh: PackedTriMesh::Proto =
            pack_triangle_mesh(&*smol::block_on(mesh.eval(&lazy_cache))?);

        mesh.flatten_into(&mut File::create("baked/derp.mesh")?);
        let unique_images: Vec<Lazy<GpuImage::Proto>> = mesh
            .maps
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let loaded_images = unique_images
            .iter()
            .cloned()
            .map(|img| smol::spawn(img.eval(&lazy_cache)));

        let loaded_images: Vec<Arc<GpuImage::Proto>> =
            smol::block_on(futures::future::try_join_all(loaded_images))
                .expect("Failed to load mesh images");

        for (image, handle) in loaded_images.into_iter().zip(unique_images) {
            image.flatten_into(&mut File::create(format!(
                "baked/{:8.8x}.image",
                handle.identity()
            ))?);
        }
    }

    /*{
        let image = GpuImage::Proto {
            format: slingshot::ash::vk::Format::R32G32B32A32_SFLOAT,
            extent: [128, 128, 1],
            mips: vec![vec![1], vec![1, 2], vec![1, 2, 3]],
        };

        let mut file = File::create("baked/derp.image")?;
        image.flatten_into(&mut file);
    }*/

    Ok(())
}
