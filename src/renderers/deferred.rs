use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

#[allow(clippy::too_many_arguments)]
pub fn light_gbuffer(
    rg: &mut RenderGraph,
    gbuffer: &rg::Handle<Image>,
    depth: &rg::Handle<Image>,
    sun_shadow_mask: &rg::Handle<Image>,
    ssgi: &rg::Handle<Image>,
    rtr: &rg::Handle<Image>,
    rtdgi: &rg::Handle<Image>,
    temporal_output: &mut rg::Handle<Image>,
    output: &mut rg::Handle<Image>,
    csgi_volume: &super::csgi::CsgiVolume,
    sky_cube: &rg::Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
) {
    SimpleRenderPass::new_compute(
        rg.add_pass("light gbuffer"),
        "/assets/shaders/light_gbuffer.hlsl",
    )
    .read(gbuffer)
    .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
    .read(sun_shadow_mask)
    .read(ssgi)
    .read(rtr)
    .read(rtdgi)
    .write(temporal_output)
    .write(output)
    .read(&csgi_volume.direct_cascade0)
    .read(&csgi_volume.indirect_cascade0)
    .read(sky_cube)
    .constants((gbuffer.desc().extent_inv_extent_2d(),))
    .raw_descriptor_set(1, bindless_descriptor_set)
    .dispatch(gbuffer.desc().extent);
}
