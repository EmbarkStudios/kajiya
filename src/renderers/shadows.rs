use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration},
};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

pub fn trace_sun_shadow_mask(
    rg: &mut RenderGraph,
    depth_img: &rg::Handle<Image>,
    tlas: &rg::Handle<RayTracingAcceleration>,
) -> rg::Handle<Image> {
    let mut output_img = rg.create(depth_img.desc().format(vk::Format::R8_UNORM));

    SimpleRenderPass::new_rt(
        rg.add_pass("trace shadow mask"),
        "/assets/shaders/rt/trace_sun_shadow_mask.rgen.hlsl",
        &["/assets/shaders/rt/shadow.rmiss.hlsl"],
        &[],
    )
    .read_aspect(&depth_img, vk::ImageAspectFlags::DEPTH)
    .write(&mut output_img)
    .trace_rays(tlas, output_img.desc().extent);

    output_img
}
