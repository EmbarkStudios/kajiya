use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg, GetOrCreateTemporal, SimpleRenderPass};

use super::GbufferDepth;

pub fn calculate_reprojection_map(
    rg: &mut rg::TemporalRenderGraph,
    gbuffer_depth: &GbufferDepth,
    velocity_img: &rg::Handle<Image>,
) -> rg::Handle<Image> {
    //let mut output_tex = rg.create(depth.desc().format(vk::Format::R16G16B16A16_SFLOAT));
    //let mut output_tex = rg.create(depth.desc().format(vk::Format::R32G32B32A32_SFLOAT));
    let mut output_tex = rg.create(
        gbuffer_depth
            .depth
            .desc()
            .format(vk::Format::R16G16B16A16_SNORM),
    );

    let mut prev_depth = rg
        .get_or_create_temporal(
            "reprojection.prev_depth",
            gbuffer_depth
                .depth
                .desc()
                .format(vk::Format::R32_SFLOAT)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
        )
        .unwrap();

    SimpleRenderPass::new_compute(
        rg.add_pass("reprojection map"),
        "/shaders/calculate_reprojection_map.hlsl",
    )
    .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
    .read(&gbuffer_depth.geometric_normal)
    .read(&prev_depth)
    .read(velocity_img)
    .write(&mut output_tex)
    .constants(output_tex.desc().extent_inv_extent_2d())
    .dispatch(output_tex.desc().extent);

    SimpleRenderPass::new_compute_rust(
        rg.add_pass("copy depth"),
        "copy_depth_to_r::copy_depth_to_r_cs",
    )
    .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
    .write(&mut prev_depth)
    .dispatch(prev_depth.desc().extent);

    output_tex
}
