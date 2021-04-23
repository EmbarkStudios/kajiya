use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

use super::GbufferDepth;

#[allow(clippy::too_many_arguments)]
pub fn light_gbuffer(
    rg: &mut RenderGraph,
    gbuffer_depth: &GbufferDepth,
    shadow_mask: &rg::Handle<Image>,
    ssgi: &rg::Handle<Image>,
    rtr: &rg::Handle<Image>,
    rtdgi: &rg::Handle<Image>,
    temporal_output: &mut rg::Handle<Image>,
    output: &mut rg::Handle<Image>,
    csgi_volume: &super::csgi::CsgiVolume,
    sky_cube: &rg::Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
    debug_shading_mode: usize,
) {
    SimpleRenderPass::new_compute(rg.add_pass("light gbuffer"), "/shaders/light_gbuffer.hlsl")
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(shadow_mask)
        .read(ssgi)
        .read(rtr)
        .read(rtdgi)
        .write(temporal_output)
        .write(output)
        .read(&csgi_volume.direct_cascade0)
        .read(&csgi_volume.indirect_cascade0)
        .read(sky_cube)
        .constants((
            gbuffer_depth.gbuffer.desc().extent_inv_extent_2d(),
            debug_shading_mode as u32,
        ))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .dispatch(gbuffer_depth.gbuffer.desc().extent);
}
