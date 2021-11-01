use async_channel::unbounded;
use async_executor::Executor;
use easy_parallel::Parallel;
use glam::Quat;
use kajiya_asset::mesh::{pack_triangle_mesh, GpuImage, LoadGltfScene, PackedTriMesh};
use smol::future;
use std::{collections::HashSet, fs::File, path::PathBuf};

use turbosloth::*;

use anyhow::Result;
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
    let lazy_cache = LazyCache::create();

    let opt = Opt::from_args();
    std::fs::create_dir_all("baked")?;

    {
        println!("Loading {:?}...", opt.scene);

        let mesh = LoadGltfScene {
            path: opt.scene,
            scale: opt.scale,
            //rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            rotation: Quat::IDENTITY,
        }
        .into_lazy();

        let mesh = &*smol::block_on(mesh.eval(&lazy_cache))?;

        println!("Packing the mesh...");
        let mesh: PackedTriMesh::Proto = pack_triangle_mesh(mesh);

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

        let ex = &Executor::new();
        let (signal, shutdown) = unbounded::<()>();

        // Prepare tasks for processing all images
        let lazy_cache = &lazy_cache;
        let images = unique_images.iter().cloned().map(|img| async move {
            let loaded = img.eval(lazy_cache).await?;

            loaded.flatten_into(&mut File::create(format!(
                "baked/{:8.8x}.image",
                img.identity()
            ))?);

            //println!("Wrote baked/{:8.8x}.image", img.identity());

            anyhow::Result::<()>::Ok(())
        });

        // Now spawn them onto the executor
        let images = images.map(|task| ex.spawn(task));
        let image_count = images.len();

        if image_count > 0 {
            // A task to join them all
            let all_images = futures::future::try_join_all(images);

            println!("Processing {} images...", image_count);

            // Now spawn threads for the executor and run it to completion
            Parallel::new()
                .each(0..num_cpus::get(), |_| {
                    future::block_on(ex.run(shutdown.recv()))
                })
                .finish(|| {
                    future::block_on(async {
                        all_images.await.expect("Failed to load mesh images");
                        drop(signal);
                    })
                });
        }

        println!("Done.");
    }

    Ok(())
}
