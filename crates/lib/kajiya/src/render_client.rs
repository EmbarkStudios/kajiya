use std::sync::Arc;

use crate::{image_cache::UploadGpuImage, world_renderer::WorldRenderer};
use kajiya_asset::{
    image::LoadImage,
    mesh::{TexGamma, TexParams},
};
use kajiya_backend::{
    ash::vk,
    vk_sync::{self, AccessType},
    vulkan::{image::*, RenderBackend},
};
use kajiya_rg::{self as rg};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use turbosloth::*;

#[derive(Default)]
pub struct UiRenderer {
    pub ui_frame: Option<(UiRenderCallback, Arc<Image>)>,
}

pub type UiRenderCallback = Box<dyn FnOnce(vk::CommandBuffer) + 'static>;

impl UiRenderer {
    pub fn prepare_render_graph(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
    ) -> rg::ExportedHandle<Image> {
        let ui_img = self.render_ui(rg);
        rg.export(
            ui_img,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        )
    }

    fn render_ui(&mut self, rg: &mut rg::RenderGraph) -> rg::Handle<Image> {
        if let Some((ui_renderer, image)) = self.ui_frame.take() {
            let mut ui_tex = rg.import(image, AccessType::Nothing);
            let mut pass = rg.add_pass("render ui");

            pass.raster(&mut ui_tex, AccessType::ColorAttachmentWrite);
            pass.render(move |api| ui_renderer(api.cb.raw));

            ui_tex
        } else {
            let mut blank_img = rg.create(ImageDesc::new_2d(vk::Format::R8G8B8A8_UNORM, [1, 1]));
            crate::renderers::imageops::clear_color(rg, &mut blank_img, [0.0f32; 4]);
            blank_img
        }
    }
}

pub fn create_default_world_renderer(
    backend: &RenderBackend,
    lazy_cache: &Arc<LazyCache>,
) -> anyhow::Result<WorldRenderer> {
    let mut world_renderer = WorldRenderer::new(backend)?;

    // BINDLESS_LUT_BRDF_FG
    world_renderer.add_image_lut(crate::lut_renderers::BrdfFgLutComputer, 0);

    {
        let image = LoadImage::new("assets/images/bluenoise/256_256/LDR_RGBA_0.png")?.into_lazy();
        let blue_noise_img = smol::block_on(
            UploadGpuImage {
                image,
                params: TexParams {
                    gamma: TexGamma::Linear,
                    use_mips: false,
                },
                device: backend.device.clone(),
            }
            .into_lazy()
            .eval(lazy_cache),
        )
        .unwrap();

        let handle = world_renderer.add_image(blue_noise_img);

        // BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0
        assert_eq!(handle.0, 1);
    }

    Ok(world_renderer)
}
