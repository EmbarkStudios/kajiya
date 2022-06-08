use crate::{self as rg, RenderGraph};
use kajiya_backend::{ash::vk, vk_sync::AccessType, vulkan::image::*};

pub fn clear_depth(rg: &mut RenderGraph, img: &mut rg::Handle<Image>) {
    let mut pass = rg.add_pass("clear depth");
    let output_ref = pass.write(img, AccessType::TransferWrite);

    pass.render(move |api| {
        let raw_device = &api.device().raw;
        let cb = api.cb;

        let image = api.resources.image(output_ref);

        unsafe {
            raw_device.cmd_clear_depth_stencil_image(
                cb.raw,
                image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue {
                    depth: 0f32,
                    stencil: 0,
                },
                std::slice::from_ref(&vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                }),
            );
        }

        Ok(())
    });
}

pub fn clear_color(rg: &mut RenderGraph, img: &mut rg::Handle<Image>, clear_color: [f32; 4]) {
    let mut pass = rg.add_pass("clear color");
    let output_ref = pass.write(img, AccessType::TransferWrite);

    pass.render(move |api| {
        let raw_device = &api.device().raw;
        let cb = api.cb;

        let image = api.resources.image(output_ref);

        unsafe {
            raw_device.cmd_clear_color_image(
                cb.raw,
                image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: clear_color,
                },
                std::slice::from_ref(&vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                }),
            );
        }

        Ok(())
    });
}
