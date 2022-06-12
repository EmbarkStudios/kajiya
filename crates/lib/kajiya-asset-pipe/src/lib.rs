use async_channel::unbounded;
use async_executor::Executor;
use easy_parallel::Parallel;
use glam::Quat;
use kajiya_asset::mesh::{pack_triangle_mesh, GpuImage, LoadGltfScene, PackedTriMesh};
use smol::future;
use std::{collections::HashSet, fs::File, path::PathBuf};

use turbosloth::*;

use anyhow::Result;

pub struct MeshAssetProcessParams {
    pub path: PathBuf,
    pub output_name: String,
    pub scale: f32,
}

pub fn process_mesh_asset(opt: MeshAssetProcessParams) -> Result<()> {
    let lazy_cache = LazyCache::create();

    std::fs::create_dir_all("cache")?;

    {
        println!("Loading {:?}...", opt.path);

        let mesh = LoadGltfScene {
            path: opt.path,
            scale: opt.scale,
            //rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            rotation: Quat::IDENTITY,
        }
        .into_lazy();

        let mesh = &*smol::block_on(mesh.eval(&lazy_cache))?;

        println!("Packing the mesh...");
        let mesh: PackedTriMesh::Proto = pack_triangle_mesh(mesh);

        mesh.flatten_into(&mut File::create(format!(
            "cache/{}.mesh",
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
            let img_dst = PathBuf::from(format!("cache/{:8.8x}.image", img.identity()));

            match File::create(&img_dst) {
                Ok(mut file) => loaded.flatten_into(&mut file),
                Err(err) => {
                    if img_dst.exists() {
                        log::info!("Could not create {:?}; ignoring", img_dst);
                    } else {
                        anyhow::anyhow!(err);
                    }
                }
            };

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
