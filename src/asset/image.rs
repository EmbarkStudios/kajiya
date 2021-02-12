use std::{path::PathBuf, sync::Arc};

use image::GenericImageView as _;
use slingshot::file::LoadFile;
use turbosloth::*;

#[derive(Clone, Hash)]
pub struct LoadImage {
    pub path: PathBuf,
}

pub struct RawRgba8Image {
    pub data: Vec<u8>,
    pub dimensions: [u32; 2],
}

#[async_trait]
impl LazyWorker for LoadImage {
    type Output = anyhow::Result<RawRgba8Image>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let file: Arc<Vec<u8>> = LoadFile {
            path: self.path.clone(),
        }
        .into_lazy()
        .eval(&ctx)
        .await?;

        let image = image::load_from_memory(file.as_slice())?;
        let image_dimensions = image.dimensions();
        log::info!("Loaded image: {:?} {:?}", image_dimensions, image.color());

        let image = image.to_rgba();

        Ok(RawRgba8Image {
            data: image.into_raw(),
            dimensions: [image_dimensions.0, image_dimensions.1],
        })
    }
}
