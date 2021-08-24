use super::{GbufferDepth, PingPongTemporalResource};
use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg, SimpleRenderPass, TemporalRenderGraph};

pub struct ShadowDenoiseRenderer {
    accum: PingPongTemporalResource,
    moments: PingPongTemporalResource,
}

impl Default for ShadowDenoiseRenderer {
    fn default() -> Self {
        Self {
            accum: PingPongTemporalResource::new("shadow_denoise_accum"),
            moments: PingPongTemporalResource::new("shadow_denoise_moments"),
        }
    }
}

impl ShadowDenoiseRenderer {
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        shadow_mask: &rg::Handle<Image>,
        reprojection_map: &rg::Handle<Image>,
    ) -> rg::ReadOnlyHandle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let bitpacked_shadow_mask_extent = gbuffer_desc.div_up_extent([8, 4, 1]).extent_2d();
        let mut bitpacked_shadows_image = rg.create(ImageDesc::new_2d(
            vk::Format::R32_UINT,
            bitpacked_shadow_mask_extent,
        ));

        SimpleRenderPass::new_compute(
            rg.add_pass("shadow bitpack"),
            "/shaders/shadow_denoise/bitpack_shadow_mask.hlsl",
        )
        .read(shadow_mask)
        .write(&mut bitpacked_shadows_image)
        .constants((
            gbuffer_desc.extent_inv_extent_2d(),
            bitpacked_shadow_mask_extent,
        ))
        .dispatch([
            bitpacked_shadow_mask_extent[0] * 2,
            bitpacked_shadow_mask_extent[1],
            1,
        ]);

        let (mut moments_image, prev_moments_image) = self.moments.get_output_and_history(
            rg,
            gbuffer_desc
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let spatial_image_desc =
            ImageDesc::new_2d(vk::Format::R16G16_SFLOAT, gbuffer_desc.extent_2d());

        let (mut accum_image, prev_accum_image) = self.accum.get_output_and_history(
            rg,
            spatial_image_desc.usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
        );

        let mut spatial_input_image = rg.create(spatial_image_desc);
        let mut metadata_image = rg.create(ImageDesc::new_2d(
            vk::Format::R32_UINT,
            bitpacked_shadow_mask_extent,
        ));

        SimpleRenderPass::new_compute(
            rg.add_pass("shadow temporal"),
            "/shaders/shadow_denoise/megakernel.hlsl",
        )
        .read(shadow_mask)
        .read(&bitpacked_shadows_image)
        .read(&prev_moments_image)
        .read(&prev_accum_image)
        .read(reprojection_map)
        .write(&mut moments_image)
        .write(&mut spatial_input_image)
        .write(&mut metadata_image)
        .constants((
            gbuffer_desc.extent_inv_extent_2d(),
            bitpacked_shadow_mask_extent,
        ))
        .dispatch(gbuffer_desc.extent);

        let mut temp = rg.create(spatial_image_desc);
        Self::filter_spatial(
            rg,
            1,
            &spatial_input_image,
            &mut accum_image,
            &metadata_image,
            gbuffer_depth,
            bitpacked_shadow_mask_extent,
        );

        Self::filter_spatial(
            rg,
            2,
            &accum_image,
            &mut temp,
            &metadata_image,
            gbuffer_depth,
            bitpacked_shadow_mask_extent,
        );

        Self::filter_spatial(
            rg,
            4,
            &temp,
            &mut spatial_input_image,
            &metadata_image,
            gbuffer_depth,
            bitpacked_shadow_mask_extent,
        );

        spatial_input_image.into()
    }

    fn filter_spatial(
        rg: &mut TemporalRenderGraph,
        step_size: u32,
        input_image: &rg::Handle<Image>,
        output_image: &mut rg::Handle<Image>,
        metadata_image: &rg::Handle<Image>,
        gbuffer_depth: &GbufferDepth,
        bitpacked_shadow_mask_extent: [u32; 2],
    ) {
        SimpleRenderPass::new_compute(
            rg.add_pass("shadow spatial"),
            "/shaders/shadow_denoise/spatial_filter.hlsl",
        )
        .read(input_image)
        .read(metadata_image)
        .read(&gbuffer_depth.geometric_normal)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .write(output_image)
        .constants((
            output_image.desc().extent_inv_extent_2d(),
            bitpacked_shadow_mask_extent,
            step_size,
        ))
        .dispatch(output_image.desc().extent);
    }
}
