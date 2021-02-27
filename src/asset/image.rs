use std::{path::PathBuf, sync::Arc};

use image::{imageops::FilterType, DynamicImage, GenericImageView as _};
use slingshot::{ash::vk, file::LoadFile, ImageDesc};
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

#[derive(Clone, Hash)]
pub struct CreateGpuImage {
    pub image: Lazy<RawRgba8Image>,
    pub params: super::mesh::TexParams,
}

#[async_trait]
impl LazyWorker for CreateGpuImage {
    type Output = anyhow::Result<super::mesh::GpuImage::Proto>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let src = self.image.eval(&ctx).await?;

        let format = match self.params.gamma {
            crate::asset::mesh::TexGamma::Linear => vk::Format::R8G8B8A8_UNORM,
            crate::asset::mesh::TexGamma::Srgb => vk::Format::R8G8B8A8_SRGB,
        };

        let mut desc =
            ImageDesc::new_2d(format, src.dimensions).usage(vk::ImageUsageFlags::SAMPLED);

        let mut mips: Vec<Vec<u8>> = vec![src.data.clone()];
        if self.params.use_mips {
            desc = desc.all_mip_levels();

            let mut image = image::DynamicImage::ImageRgba8(
                image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                    src.dimensions[0],
                    src.dimensions[1],
                    src.data.clone(),
                )
                .unwrap(),
            );

            let downsample = |image: &DynamicImage| {
                // TODO: gamma-correct resize
                image.resize_exact(
                    (image.dimensions().0 / 2).max(1),
                    (image.dimensions().1 / 2).max(1),
                    FilterType::Lanczos3,
                )
            };

            image = downsample(&image);

            for _ in 1..desc.mip_levels {
                let next = downsample(&image);
                let mip = std::mem::replace(&mut image, next);
                mips.push(mip.into_rgba8().into_raw());
            }
        }

        Ok(super::mesh::GpuImage::Proto {
            format,
            extent: [src.dimensions[0], src.dimensions[1], 1],
            mips,
        })

        //self.device.create_image(desc, initial_data)
    }
}
