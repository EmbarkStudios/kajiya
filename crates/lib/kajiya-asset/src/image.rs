use std::path::PathBuf;

use bytes::Bytes;
use image::{imageops::FilterType, DynamicImage, GenericImageView as _, ImageBuffer, Rgba};
use intel_tex_2::{bc5, bc7};
use kajiya_backend::{ash::vk, file::LoadFile, ImageDesc};
use turbosloth::*;

use crate::mesh::TexCompressionMode;

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum ImageSource {
    File(PathBuf),
    Memory(Bytes),
}

pub struct RawRgba8Image {
    pub data: Bytes,
    pub dimensions: [u32; 2],
}

#[allow(clippy::large_enum_variant)]
pub enum RawImage {
    Rgba8(RawRgba8Image),
    Dds(ddsfile::Dds),
}

#[derive(Clone, Hash)]
pub enum LoadImage {
    Lazy(Lazy<Bytes>),
    Immediate(Bytes),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BcMode {
    Bc5,
    Bc7,
}

impl BcMode {
    fn block_bytes(self) -> usize {
        match self {
            BcMode::Bc5 => 16,
            BcMode::Bc7 => 16,
        }
    }
}

impl LoadImage {
    pub fn from_path<P: Into<PathBuf>>(path: P) -> anyhow::Result<Self> {
        Self::new(&ImageSource::File(path.into()))
    }

    pub fn new(source: &ImageSource) -> anyhow::Result<Self> {
        match source {
            ImageSource::File(path) => Ok(Self::Lazy(LoadFile::new(path)?.into_lazy())),
            ImageSource::Memory(bytes) => Ok(Self::Immediate(bytes.clone())),
        }
    }
}

#[async_trait]
impl LazyWorker for LoadImage {
    type Output = anyhow::Result<RawImage>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let bytes: Bytes = match self {
            // Note: `Bytes` does internal reference counting, so this clone is cheap
            LoadImage::Lazy(bytes) => Bytes::clone(bytes.eval(&ctx).await?.as_ref()),
            LoadImage::Immediate(bytes) => bytes,
        };

        if let Ok(dds) = ddsfile::Dds::read(&mut std::io::Cursor::new(&bytes)) {
            log::info!(
                "Loaded DDS image: {}x{}x{} {}",
                dds.get_width(),
                dds.get_height(),
                dds.get_depth(),
                dds.get_dxgi_format().map_or_else(
                    || format!("d3d:{:?}", dds.get_d3d_format()),
                    |fmt| format!("dxgi:{:?}", fmt),
                )
            );

            Ok(RawImage::Dds(dds))
        } else {
            let image = image::load_from_memory(&bytes)?;
            let image_dimensions = image.dimensions();
            log::info!("Loaded image: {:?} {:?}", image_dimensions, image.color());

            let image = image.to_rgba8();

            Ok(RawImage::Rgba8(RawRgba8Image {
                data: image.into_raw().into(),
                dimensions: [image_dimensions.0, image_dimensions.1],
            }))
        }
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
    type Output = anyhow::Result<RawImage>;

    async fn run(self, _ctx: RunContext) -> Self::Output {
        Ok(RawImage::Rgba8(RawRgba8Image {
            data: Bytes::from(self.values.to_vec()),
            dimensions: [1, 1],
        }))
    }
}

#[derive(Clone, Hash)]
pub struct CreateGpuImage {
    pub image: Lazy<RawImage>,
    pub params: super::mesh::TexParams,
}

impl CreateGpuImage {
    fn process_rgba8(&self, src: &RawRgba8Image) -> anyhow::Result<super::mesh::GpuImage::Proto> {
        let mut format = match self.params.gamma {
            crate::mesh::TexGamma::Linear => vk::Format::R8G8B8A8_UNORM,
            crate::mesh::TexGamma::Srgb => vk::Format::R8G8B8A8_SRGB,
        };

        let mut image = image::DynamicImage::ImageRgba8(
            image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                src.dimensions[0],
                src.dimensions[1],
                src.data.to_vec(),
            )
            .unwrap(),
        );

        let should_compress = self.params.compression != TexCompressionMode::None
            && image.width() >= 4
            && image.height() >= 4;

        const MAX_SIZE: u32 = 2048;

        if image.dimensions().0 > MAX_SIZE || image.dimensions().1 > MAX_SIZE {
            image = image.resize_exact(
                image.dimensions().0.min(MAX_SIZE),
                image.dimensions().1.min(MAX_SIZE),
                FilterType::Lanczos3,
            );
        }

        let mut desc = ImageDesc::new_2d(format, [image.dimensions().0, image.dimensions().1])
            .usage(vk::ImageUsageFlags::SAMPLED);

        let mut compress = |mip: ImageBuffer<Rgba<u8>, Vec<u8>>| -> Vec<u8> {
            let block_count = intel_tex_2::divide_up_by_multiple(mip.width() * mip.height(), 16);

            let needs_alpha =
                self.params.compression.supports_alpha() && mip.pixels().any(|px| px.0[3] != 255);

            let bc_mode = match self.params.compression {
                TexCompressionMode::None => unreachable!(),
                TexCompressionMode::Rgba => BcMode::Bc7,
                TexCompressionMode::Rg => BcMode::Bc5,
            };

            let block_bytes = bc_mode.block_bytes();

            let surface = intel_tex_2::RgbaSurface {
                width: mip.width(),
                height: mip.height(),
                stride: mip.width() * 4,
                data: &mip,
            };

            let mut compressed_bytes = vec![0u8; block_count as usize * block_bytes];

            log::info!("Compressing to {:?}...", bc_mode);
            match bc_mode {
                BcMode::Bc5 => {
                    format = match self.params.gamma {
                        crate::mesh::TexGamma::Linear => vk::Format::BC5_UNORM_BLOCK,
                        crate::mesh::TexGamma::Srgb => unimplemented!(),
                    };

                    bc5::compress_blocks_into(&surface, &mut compressed_bytes)
                }
                BcMode::Bc7 => {
                    format = match self.params.gamma {
                        crate::mesh::TexGamma::Linear => vk::Format::BC7_UNORM_BLOCK,
                        crate::mesh::TexGamma::Srgb => vk::Format::BC7_SRGB_BLOCK,
                    };

                    let settings = if needs_alpha {
                        bc7::alpha_basic_settings()
                    } else {
                        bc7::opaque_basic_settings()
                    };

                    bc7::compress_blocks_into(&settings, &surface, &mut compressed_bytes);
                }
            }

            compressed_bytes
        };

        let swizzle = |mip: &mut ImageBuffer<Rgba<u8>, Vec<u8>>| {
            if let Some(swizzle) = self.params.channel_swizzle {
                for px in mip.pixels_mut() {
                    px.0[0] = px.0[swizzle[0]];
                    px.0[1] = px.0[swizzle[1]];
                    px.0[2] = px.0[swizzle[2]];
                    px.0[3] = px.0[swizzle[3]];
                }
            }
        };

        let min_img_dim = if should_compress { 4 } else { 1 };

        let round_up_to_block = |x: u32| -> u32 {
            (((x + min_img_dim - 1) / min_img_dim) * min_img_dim).max(min_img_dim)
        };

        let mut process_mip = |mip: DynamicImage| -> Vec<u8> {
            let mip = if mip.width() % min_img_dim != 0 || mip.height() % min_img_dim != 0 {
                let width = round_up_to_block(mip.width());
                let height = round_up_to_block(mip.height());
                mip.resize_exact(width, height, FilterType::Lanczos3)
            } else {
                mip
            };

            let mut mip = mip.into_rgba8();

            swizzle(&mut mip);

            if should_compress {
                compress(mip)
            } else {
                mip.into_raw()
            }
        };

        let mips: Vec<Vec<u8>> = if self.params.use_mips {
            desc = desc.all_mip_levels();

            let downsample = |image: &DynamicImage| {
                // TODO: gamma-correct resize
                image.resize_exact(
                    round_up_to_block(image.dimensions().0 / 2),
                    round_up_to_block(image.dimensions().1 / 2),
                    FilterType::Lanczos3,
                )
            };

            let mut mips;
            image = {
                let next = downsample(&image);
                mips = vec![process_mip(image)];
                next
            };

            for _ in 1..desc.mip_levels {
                let next = downsample(&image);
                let mip = std::mem::replace(&mut image, next);
                mips.push(process_mip(mip));
            }

            mips
        } else {
            vec![process_mip(image)]
        };

        Ok(super::mesh::GpuImage::Proto {
            format,
            extent: desc.extent,
            mips,
        })
    }

    fn process_dds(&self, dds: &ddsfile::Dds) -> anyhow::Result<super::mesh::GpuImage::Proto> {
        if dds_util::get_pitch(dds, dds.get_width()).is_none() {
            anyhow::bail!("Not pitch available for DDS image");
        }

        let dds_data = dds.get_data(0).unwrap();
        let mut byte_offset = 0usize;

        // 1 for regular, 4 for BC
        let pitch_height = dds.get_pitch_height();

        let mips: Vec<Vec<u8>> = (0..dds.get_num_mipmap_levels())
            .map(|mip| -> Vec<u8> {
                let width = (dds.get_width() >> mip).max(pitch_height);
                let height = (dds.get_height() >> mip).max(pitch_height);
                let pitch = dds_util::get_pitch(dds, width).unwrap();

                let mip_size_bytes = dds_util::get_texture_size(pitch, pitch_height, height, 1);

                let mip_data = &dds_data[byte_offset..byte_offset + mip_size_bytes];

                byte_offset += mip_size_bytes;

                mip_data.to_owned()
            })
            .collect();

        assert_eq!(byte_offset, dds_data.len());

        let format = match dds.get_dxgi_format() {
            Some(ddsfile::DxgiFormat::BC1_UNorm_sRGB) => vk::Format::BC1_RGB_SRGB_BLOCK,
            Some(ddsfile::DxgiFormat::BC3_UNorm) => vk::Format::BC3_UNORM_BLOCK,
            Some(ddsfile::DxgiFormat::BC3_UNorm_sRGB) => vk::Format::BC3_SRGB_BLOCK,
            Some(ddsfile::DxgiFormat::BC5_UNorm) => vk::Format::BC5_UNORM_BLOCK,
            Some(ddsfile::DxgiFormat::BC5_SNorm) => vk::Format::BC5_SNORM_BLOCK,
            _ => todo!(
                "DDS format dxgi:{:?} d3d:{:?} not supported yet",
                dds.get_dxgi_format(),
                dds.get_d3d_format()
            ),
        };

        Ok(super::mesh::GpuImage::Proto {
            format,
            extent: [dds.get_width(), dds.get_height(), dds.get_depth()],
            mips,
        })
    }
}

// From `ddsfile`, with some modifications
mod dds_util {
    pub fn get_texture_size(pitch: u32, pitch_height: u32, height: u32, depth: u32) -> usize {
        let row_height = (height + (pitch_height - 1)) / pitch_height;
        pitch as usize * row_height as usize * depth as usize
    }

    pub fn get_pitch(dds: &ddsfile::Dds, width: u32) -> Option<u32> {
        // Try format first
        if let Some(format) = dds.get_format() {
            if let Some(pitch) = format.get_pitch(width) {
                return Some(pitch);
            }
        }

        // Then try to calculate it ourselves
        if let Some(bpp) = dds.get_bits_per_pixel() {
            return Some((bpp * width + 7) / 8);
        }
        None
    }
}

#[async_trait]
impl LazyWorker for CreateGpuImage {
    type Output = anyhow::Result<super::mesh::GpuImage::Proto>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let src = self.image.eval(&ctx).await?;

        match &*src {
            RawImage::Rgba8(src) => self.process_rgba8(src),
            RawImage::Dds(src) => self.process_dds(src),
        }
    }
}
