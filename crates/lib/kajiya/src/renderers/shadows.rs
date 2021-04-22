use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration},
};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

use super::GbufferDepth;

pub fn trace_sun_shadow_mask(
    rg: &mut RenderGraph,
    gbuffer_depth: &GbufferDepth,
    tlas: &rg::Handle<RayTracingAcceleration>,
    bindless_descriptor_set: vk::DescriptorSet,
) -> rg::Handle<Image> {
    let mut output_img = rg.create(gbuffer_depth.depth.desc().format(vk::Format::R8_UNORM));

    SimpleRenderPass::new_rt(
        rg.add_pass("trace shadow mask"),
        "/shaders/rt/trace_sun_shadow_mask.rgen.hlsl",
        &[
            // Duplicated because `rt.hlsl` hardcodes miss index to 1
            "/shaders/rt/shadow.rmiss.hlsl",
            "/shaders/rt/shadow.rmiss.hlsl",
        ],
        &[],
    )
    .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
    .read(&gbuffer_depth.geometric_normal)
    .write(&mut output_img)
    .raw_descriptor_set(1, bindless_descriptor_set)
    .trace_rays(tlas, output_img.desc().extent);

    output_img
}
