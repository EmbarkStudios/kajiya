use kajiya_backend::{
    ash::vk,
    backend::image::*,
    rg::{self, SimpleRenderPass},
};

pub fn render_sky_cube(rg: &mut rg::RenderGraph) -> rg::Handle<Image> {
    let width = 32;
    let mut sky_tex = rg.create(ImageDesc::new_cube(vk::Format::R16G16B16A16_SFLOAT, width));

    SimpleRenderPass::new_compute(
        rg.add_pass("sky cube"),
        "/assets/shaders/sky/comp_cube.hlsl",
    )
    .write_view(
        &mut sky_tex,
        ImageViewDesc::builder().view_type(vk::ImageViewType::TYPE_2D_ARRAY),
    )
    .dispatch([width, width, 6]);

    sky_tex
}

pub fn convolve_cube(rg: &mut rg::RenderGraph, input: &rg::Handle<Image>) -> rg::Handle<Image> {
    let width = 16u32;
    let mut sky_tex = rg.create(ImageDesc::new_cube(vk::Format::R16G16B16A16_SFLOAT, width));

    SimpleRenderPass::new_compute(
        rg.add_pass("convolve sky"),
        "/assets/shaders/convolve_cube.hlsl",
    )
    .read(input)
    .write_view(
        &mut sky_tex,
        ImageViewDesc::builder().view_type(vk::ImageViewType::TYPE_2D_ARRAY),
    )
    .constants(width)
    .dispatch([width, width, 6]);

    sky_tex
}
