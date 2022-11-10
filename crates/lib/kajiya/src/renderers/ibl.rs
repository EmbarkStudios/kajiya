use anyhow::Context;
use exr::prelude::{self as exrs, ReadChannels as _, ReadLayers as _};
use half::f16;
use std::{fs::File, io::BufReader, path::Path, sync::Arc};

use kajiya_backend::{
    ash::vk::{self, ImageUsageFlags},
    vulkan::image::*,
};
use kajiya_rg::{self as rg, SimpleRenderPass};

#[derive(Default)]
pub struct IblRenderer {
    image: Option<ImageRgba16f>,
    texture: Option<Arc<Image>>,
}

impl IblRenderer {
    pub fn unload_image(&mut self) {
        self.image = None;
        // TODO: deallocate
        self.texture = None;
    }

    pub fn load_image(&mut self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let img = load_image(path.as_ref())?;

        self.image = Some(img);

        // Force re-creation of the texture
        // TODO: deallocate the old one 😅
        self.texture = None;

        Ok(())
    }

    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
    ) -> Option<rg::ReadOnlyHandle<Image>> {
        if self.texture.is_none() {
            const PIXEL_BYTES: u32 = 8;

            if let Some(image) = self.image.take() {
                self.texture = Some(Arc::new(
                    rg.device()
                        .create_image(
                            ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, image.size)
                                .usage(ImageUsageFlags::SAMPLED),
                            vec![ImageSubResourceData {
                                data: bytemuck::checked::cast_slice(image.data.as_slice()),
                                row_pitch: (image.size[0] * PIXEL_BYTES) as usize,
                                slice_pitch: (image.size[0] * image.size[1] * PIXEL_BYTES) as usize,
                            }],
                        )
                        .expect("create_image"),
                ));
            }
        }

        if let Some(texture) = self.texture.clone() {
            let width = 1024u32;
            let mut cube_tex =
                rg.create(ImageDesc::new_cube(vk::Format::R16G16B16A16_SFLOAT, width));

            let texture = rg.import(
                texture,
                kajiya_backend::vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
            );

            SimpleRenderPass::new_compute(rg.add_pass("ibl cube"), "/shaders/ibl/ibl_cube.hlsl")
                .read(&texture)
                .write_view(
                    &mut cube_tex,
                    ImageViewDesc::builder().view_type(vk::ImageViewType::TYPE_2D_ARRAY),
                )
                .constants(width)
                .dispatch([width, width, 6]);

            Some(cube_tex.into())
        } else {
            None
        }
    }
}

pub struct ImageRgba16f {
    pub size: [u32; 2],
    pub data: Vec<f16>,
}

impl ImageRgba16f {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            size: [width, height],
            data: vec![f16::ZERO; (width * height * 4) as usize],
        }
    }

    fn put_pixel(&mut self, x: u32, y: u32, rgba: [f16; 4]) {
        let offset = ((y * self.size[0] + x) * 4) as usize;
        self.data[offset..offset + 4].copy_from_slice(&rgba);
    }
}

fn load_image(path: &Path) -> anyhow::Result<ImageRgba16f> {
    let ext = path
        .extension()
        .map(|ext| ext.to_string_lossy().as_ref().to_owned());

    match ext.as_deref() {
        Some("exr") => load_exr(path),
        Some("hdr") => load_hdr(path),
        _ => Err(anyhow::anyhow!("Unsupported file extension: {:?}", ext)),
    }
}

fn load_hdr(file_path: &Path) -> anyhow::Result<ImageRgba16f> {
    let f = File::open(file_path).context("failed to open specified file")?;
    let f = BufReader::new(f);
    let image = radiant::load(f).context("failed to load image data")?;

    let data: Vec<f16> = image
        .data
        .iter()
        .copied()
        .flat_map(|px| {
            [
                f16::from_f32(px.r).min(f16::MAX),
                f16::from_f32(px.g).min(f16::MAX),
                f16::from_f32(px.b).min(f16::MAX),
                f16::ONE,
            ]
            .into_iter()
        })
        .collect();

    Ok(ImageRgba16f {
        size: [image.width as _, image.height as _],
        data,
    })
}

fn load_exr(file_path: &Path) -> anyhow::Result<ImageRgba16f> {
    let reader = exrs::read()
        .no_deep_data()
        .largest_resolution_level()
        .rgb_channels(
            |resolution, _channels: &exrs::RgbChannels| -> ImageRgba16f {
                ImageRgba16f::new(resolution.width() as _, resolution.height() as _)
            },
            // set each pixel in the png buffer from the exr file
            |output, position, (r, g, b): (f16, f16, f16)| {
                output.put_pixel(
                    position.0 as _,
                    position.1 as _,
                    [r.min(f16::MAX), g.min(f16::MAX), b.min(f16::MAX), f16::ONE],
                );
            },
        )
        .first_valid_layer()
        .all_attributes();

    // an image that contains a single layer containing an png rgba buffer
    let maybe_image: Result<
        exrs::Image<exrs::Layer<exrs::SpecificChannels<ImageRgba16f, exrs::RgbChannels>>>,
        exrs::Error,
    > = reader.from_file(file_path);

    let output = maybe_image?.layer_data.channel_data.pixels;
    Ok(output)
}
