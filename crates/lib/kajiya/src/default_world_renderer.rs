use std::sync::Arc;

use crate::{image_cache::UploadGpuImage, world_renderer::WorldRenderer};
use kajiya_asset::{
    image::LoadImage,
    mesh::{TexGamma, TexParams},
};
use kajiya_backend::vulkan::RenderBackend;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use turbosloth::*;

impl WorldRenderer {
    pub fn new(
        render_extent: [u32; 2],
        temporal_upscale_extent: [u32; 2],
        backend: &RenderBackend,
        lazy_cache: &Arc<LazyCache>,
    ) -> anyhow::Result<Self> {
        let mut world_renderer = Self::new_empty(render_extent, temporal_upscale_extent, backend)?;

        // BINDLESS_LUT_BRDF_FG
        world_renderer.add_image_lut(crate::lut_renderers::BrdfFgLutComputer, 0);

        {
            let image =
                LoadImage::from_path("/images/bluenoise/256_256/LDR_RGBA_0.png")?.into_lazy();
            let blue_noise_img = smol::block_on(
                UploadGpuImage {
                    image,
                    params: TexParams {
                        gamma: TexGamma::Linear,
                        use_mips: false,
                        compression: kajiya_asset::mesh::TexCompressionMode::None,
                        channel_swizzle: None,
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

        // BINDLESS_LUT_BEZOLD_BRUCKE
        world_renderer.add_image_lut(crate::lut_renderers::BezoldBruckeLutComputer, 2);

        // Build an empty TLAS to create the resources. We'll update it at runtime.
        if backend.device.ray_tracing_enabled() {
            world_renderer.build_ray_tracing_top_level_acceleration();
        }

        Ok(world_renderer)
    }
}
