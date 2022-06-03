use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg};
use rg::{RenderGraph, SimpleRenderPass};

pub fn dof(
    rg: &mut RenderGraph,
    input: &rg::Handle<Image>,
    depth: &rg::Handle<Image>,
) -> rg::Handle<Image> {
    let mut coc = rg.create(ImageDesc::new_2d(
        vk::Format::R16_SFLOAT,
        input.desc().extent_2d(),
    ));

    let mut coc_tiles = rg.create(ImageDesc::new_2d(
        vk::Format::R16_SFLOAT,
        coc.desc().div_up_extent([8, 8, 1]).extent_2d(),
    ));

    SimpleRenderPass::new_compute(rg.add_pass("coc"), "/shaders/dof/coc.hlsl")
        .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
        .write(&mut coc)
        .write(&mut coc_tiles)
        .dispatch(coc.desc().extent);

    let mut dof = rg.create(ImageDesc::new_2d(
        vk::Format::R16G16B16A16_SFLOAT,
        input.desc().extent_2d(),
    ));

    SimpleRenderPass::new_compute(rg.add_pass("dof gather"), "/shaders/dof/gather.hlsl")
        .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
        .read(input)
        .read(&coc)
        .read(&coc_tiles)
        .write(&mut dof)
        .constants(dof.desc().extent_inv_extent_2d())
        .dispatch(dof.desc().extent);

    dof
}
