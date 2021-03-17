use std::sync::Arc;

use glam::Vec2;
use kajiya_backend::{
    ash::{version::DeviceV1_0, vk},
    backend::{buffer::*, image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
    vk_sync::{self, AccessType},
    Device,
};
use rg::GetOrCreateTemporal;

/*        let mut cascade0_integr = rg
            .get_or_create_temporal(
                "csgi.cascade0_integr",
                ImageDesc::new_3d(
                    vk::Format::R16G16B16A16_SFLOAT,
                    [
                        VOLUME_DIMS * SLICE_COUNT as u32,
                        9 * VOLUME_DIMS,
                        VOLUME_DIMS,
                    ],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();
*/

pub fn copy_depth(
    rg: &mut rg::RenderGraph,
    input: &rg::Handle<Image>,
    output: &mut rg::Handle<Image>,
) {
    let mut pass = rg.add_pass("copy depth");
    let input_ref = pass.read(input, AccessType::TransferRead);
    let output_ref = pass.write(output, AccessType::TransferWrite);

    pass.render(move |api| {
        let raw_device = &api.device().raw;
        let cb = api.cb;

        let input = api.resources.image(input_ref);
        let output = api.resources.image(output_ref);

        let input_extent = input_ref.desc().extent;

        unsafe {
            raw_device.cmd_copy_image(
                cb.raw,
                input.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                output.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy::builder()
                    .src_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1)
                            .mip_level(0)
                            .build(),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1)
                            .mip_level(0)
                            .build(),
                    )
                    .extent(vk::Extent3D {
                        width: input_extent[0],
                        height: input_extent[1],
                        depth: input_extent[2],
                    })
                    .build()],
            );
        }
    });
}

pub fn calculate_reprojection_map(
    rg: &mut rg::TemporalRenderGraph,
    depth: &rg::Handle<Image>,
) -> rg::Handle<Image> {
    let mut output_tex = rg.create(depth.desc().format(vk::Format::R16G16B16A16_SFLOAT));

    let mut prev_depth = rg
        .get_or_create_temporal(
            "reprojection.prev_depth",
            depth
                .desc()
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
        )
        .unwrap();

    SimpleRenderPass::new_compute(
        rg.add_pass("reprojection map"),
        "/assets/shaders/calculate_reprojection_map.hlsl",
    )
    .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
    .read_aspect(&prev_depth, vk::ImageAspectFlags::DEPTH)
    .write(&mut output_tex)
    .constants(output_tex.desc().extent_inv_extent_2d())
    .dispatch(output_tex.desc().extent);

    copy_depth(rg, depth, &mut prev_depth);

    output_tex
}
