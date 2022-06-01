use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

use super::{ircache::IrcacheRenderState, wrc::WrcRenderState, GbufferDepth};

#[allow(clippy::too_many_arguments)]
pub fn light_gbuffer(
    rg: &mut RenderGraph,
    gbuffer_depth: &GbufferDepth,
    shadow_mask: &rg::Handle<Image>,
    rtr: &rg::Handle<Image>,
    rtdgi: &rg::Handle<Image>,
    ircache: &mut IrcacheRenderState,
    wrc: &WrcRenderState,
    temporal_output: &mut rg::Handle<Image>,
    output: &mut rg::Handle<Image>,
    sky_cube: &rg::Handle<Image>,
    convolved_sky_cube: &rg::Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
    debug_shading_mode: usize,
    debug_show_wrc: bool,
) {
    SimpleRenderPass::new_compute(rg.add_pass("light gbuffer"), "/shaders/light_gbuffer.hlsl")
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(shadow_mask)
        .read(rtr)
        .read(rtdgi)
        .bind_mut(ircache)
        .bind(wrc)
        .write(temporal_output)
        .write(output)
        .read(sky_cube)
        .read(convolved_sky_cube)
        .constants((
            gbuffer_depth.gbuffer.desc().extent_inv_extent_2d(),
            debug_shading_mode as u32,
            debug_show_wrc as u32,
        ))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .dispatch(gbuffer_depth.gbuffer.desc().extent);
}
