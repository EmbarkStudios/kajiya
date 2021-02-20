use std::{path::PathBuf, sync::Arc};

use image::GenericImageView as _;
use slingshot::file::LoadFile;
use turbosloth::*;

pub struct RawRgba8Image {
    pub data: Vec<u8>,
    pub dimensions: [u32; 2],
}

#[derive(Clone, Hash)]
pub struct LoadImage {
    bytes: Lazy<Vec<u8>>,
}

impl LoadImage {
    pub fn new(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        Ok(Self {
            bytes: LoadFile::new(&path.into())?.into_lazy(),
        })
    }
}

#[async_trait]
impl LazyWorker for LoadImage {
    type Output = anyhow::Result<RawRgba8Image>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let file: Arc<Vec<u8>> = self.bytes.eval(&ctx).await?;

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

#[derive(Clone, Hash)]
pub struct CreatePlaceholderImage {
    values: [u8; 4],
}

impl CreatePlaceholderImage {
    pub fn new(values: [u8; 4]) -> Self {
        Self { values }
    }
}

#[async_trait]
impl LazyWorker for CreatePlaceholderImage {
    type Output = anyhow::Result<RawRgba8Image>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        Ok(RawRgba8Image {
            data: self.values.to_vec(),
            dimensions: [1, 1],
        })
    }
}
