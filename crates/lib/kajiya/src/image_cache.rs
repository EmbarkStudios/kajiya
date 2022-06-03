use std::{hash::Hash, sync::Arc};

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use kajiya_asset::{image::RawImage, mesh::TexParams};
use kajiya_backend::{ash::vk, Device, Image, ImageDesc, ImageSubResourceData};
use turbosloth::*;

#[derive(Clone)]
pub struct UploadGpuImage {
    pub image: Lazy<RawImage>,
    pub params: TexParams,
    pub device: Arc<Device>,
}

impl Hash for UploadGpuImage {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.image.hash(state);
        self.params.hash(state);
    }
}

#[async_trait]
impl LazyWorker for UploadGpuImage {
    type Output = anyhow::Result<Image>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let src = self.image.eval(&ctx).await?;
        let src = match &*src {
            RawImage::Rgba8(src) => src,
            RawImage::Dds(_) => {
                return Err(anyhow::anyhow!("UploadGpuImage does not support Dds yet"));
            }
        };

        let format = match self.params.gamma {
            kajiya_asset::mesh::TexGamma::Linear => vk::Format::R8G8B8A8_UNORM,
            kajiya_asset::mesh::TexGamma::Srgb => vk::Format::R8G8B8A8_SRGB,
        };

        let mut desc =
            ImageDesc::new_2d(format, src.dimensions).usage(vk::ImageUsageFlags::SAMPLED);

        let mut initial_data = vec![ImageSubResourceData {
            data: &src.data,
            row_pitch: src.dimensions[0] as usize * 4,
            slice_pitch: 0,
        }];
        let mut mip_levels_data = vec![];

        if self.params.use_mips {
            desc = desc.all_mip_levels();

            let mut image = image::DynamicImage::ImageRgba8(
                image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                    src.dimensions[0],
                    src.dimensions[1],
                    src.data.to_vec(),
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
                mip_levels_data.push(mip.into_rgba8().into_raw());
            }

            initial_data.extend(
                mip_levels_data
                    .iter()
                    .enumerate()
                    .map(|(level_less_1, mip)| ImageSubResourceData {
                        data: mip.as_slice(),
                        row_pitch: ((src.dimensions[0] as usize * 4) >> (level_less_1 + 1)).max(1),
                        slice_pitch: 0,
                    }),
            );
        }

        Ok(self.device.create_image(desc, initial_data)?)
    }
}
