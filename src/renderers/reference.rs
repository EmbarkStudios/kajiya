use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration},
};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

pub fn reference_path_trace(
    rg: &mut RenderGraph,
    output_img: &mut rg::Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
    tlas: &rg::Handle<RayTracingAcceleration>,
) {
    SimpleRenderPass::new_rt(
        rg.add_pass("reference pt"),
        "/assets/shaders/rt/reference_path_trace.rgen.hlsl",
        &[
            "/assets/shaders/rt/triangle.rmiss.hlsl",
            "/assets/shaders/rt/shadow.rmiss.hlsl",
        ],
        &["/assets/shaders/rt/triangle.rchit.hlsl"],
    )
    .write(output_img)
    .raw_descriptor_set(1, bindless_descriptor_set)
    .trace_rays(tlas, output_img.desc().extent);
}
