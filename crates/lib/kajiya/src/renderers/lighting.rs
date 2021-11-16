use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration},
};
use kajiya_rg::{self as rg, SimpleRenderPass};

use super::{rtr::SPATIAL_RESOLVE_OFFSETS, GbufferDepth};

pub struct LightingRenderer {}

impl LightingRenderer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for LightingRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl LightingRenderer {
    #[allow(clippy::too_many_arguments)]
    pub fn render_specular(
        &mut self,
        output_tex: &mut rg::Handle<Image>,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let mut refl0_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        // When using PDFs stored wrt to the surface area metric, their values can be tiny or giant,
        // so fp32 is necessary. The projected solid angle metric is less sensitive, but that shader
        // variant is heavier. Overall the surface area metric and fp32 combo is faster on my RTX 2080.
        let mut refl1_tex = rg.create(refl0_tex.desc().format(vk::Format::R32G32B32A32_SFLOAT));

        let mut refl2_tex = rg.create(refl0_tex.desc().format(vk::Format::R8G8B8A8_SNORM));

        SimpleRenderPass::new_rt(
            rg.add_pass("sample lights"),
            "/shaders/lighting/sample_lights.rgen.hlsl",
            &[
                "/shaders/rt/gbuffer.rmiss.hlsl",
                "/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/shaders/rt/gbuffer.rchit.hlsl"],
        )
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .write(&mut refl0_tex)
        .write(&mut refl1_tex)
        .write(&mut refl2_tex)
        .constants((gbuffer_desc.extent_inv_extent_2d(),))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, refl0_tex.desc().extent);

        let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);
        let half_depth_tex = gbuffer_depth.half_depth(rg);

        SimpleRenderPass::new_compute(
            rg.add_pass("spatial reuse lights"),
            "/shaders/lighting/spatial_reuse_lights.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&refl0_tex)
        .read(&refl1_tex)
        .read(&refl2_tex)
        .read(&*half_view_normal_tex)
        .read(&*half_depth_tex)
        .write(output_tex)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .constants((
            output_tex.desc().extent_inv_extent_2d(),
            SPATIAL_RESOLVE_OFFSETS,
        ))
        .dispatch(output_tex.desc().extent);
    }
}
