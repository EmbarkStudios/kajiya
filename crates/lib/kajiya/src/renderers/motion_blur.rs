use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

pub fn motion_blur(
    rg: &mut RenderGraph,
    input: &rg::Handle<Image>,
    depth: &rg::Handle<Image>,
    reprojection_map: &rg::Handle<Image>,
) -> rg::Handle<Image> {
    const VELOCITY_TILE_SIZE: u32 = 16;

    let mut velocity_reduced_x = rg.create(
        reprojection_map
            .desc()
            .div_up_extent([VELOCITY_TILE_SIZE, 1, 1])
            .format(vk::Format::R16G16_SFLOAT),
    );

    SimpleRenderPass::new_compute(
        rg.add_pass("velocity reduce x"),
        "/shaders/motion_blur/velocity_reduce_x.hlsl",
    )
    .read(reprojection_map)
    .write(&mut velocity_reduced_x)
    .dispatch(velocity_reduced_x.desc().extent);

    let mut velocity_reduced_y =
        rg.create(
            velocity_reduced_x
                .desc()
                .div_up_extent([1, VELOCITY_TILE_SIZE, 1]),
        );

    SimpleRenderPass::new_compute(
        rg.add_pass("velocity reduce y"),
        "/shaders/motion_blur/velocity_reduce_y.hlsl",
    )
    .read(&velocity_reduced_x)
    .write(&mut velocity_reduced_y)
    .dispatch(velocity_reduced_x.desc().extent);

    let mut velocity_dilated = rg.create(*velocity_reduced_y.desc());

    SimpleRenderPass::new_compute(
        rg.add_pass("velocity dilate"),
        "/shaders/motion_blur/velocity_dilate.hlsl",
    )
    .read(&velocity_reduced_y)
    .write(&mut velocity_dilated)
    .dispatch(velocity_dilated.desc().extent);

    let mut output = rg.create(*input.desc());

    SimpleRenderPass::new_compute(
        rg.add_pass("motion blur"),
        "/shaders/motion_blur/motion_blur.hlsl",
    )
    .read(input)
    .read(&reprojection_map)
    .read(&velocity_dilated)
    .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
    .write(&mut output)
    .constants(output.desc().extent_inv_extent_2d())
    .dispatch(output.desc().extent);

    output
}
