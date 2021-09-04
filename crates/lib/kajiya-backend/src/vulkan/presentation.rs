use super::{
    barrier::*,
    device::{CommandBuffer, Device},
    shader::{
        create_compute_pipeline, ComputePipeline, ComputePipelineDesc, DescriptorSetLayoutOpts,
    },
    swapchain::SwapchainImage,
};
use ash::vk;

pub fn create_present_compute_shader(device: &Device) -> ComputePipeline {
    create_compute_pipeline(
        device,
        &(&include_bytes!("../final_blit.spv")[..]).to_owned(),
        &ComputePipelineDesc::builder()
            .compute_entry_hlsl("main")
            .descriptor_set_opts(&[(0, DescriptorSetLayoutOpts::builder())])
            .push_constants_bytes(4 * std::mem::size_of::<u32>())
            .build()
            .unwrap(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn blit_image_to_swapchain(
    main_tex_extent: [u32; 2],
    swapchain_extent: [u32; 2],
    device: &Device,
    cb: &CommandBuffer,
    swapchain_image: &SwapchainImage,
    main_img_view: vk::ImageView,
    ui_img_view: vk::ImageView,
    present_shader: &ComputePipeline,
) {
    record_image_barrier(
        device,
        cb.raw,
        ImageBarrier::new(
            swapchain_image.image,
            vk_sync::AccessType::Present,
            vk_sync::AccessType::ComputeShaderWrite,
            vk::ImageAspectFlags::COLOR,
        )
        .with_discard(true),
    );

    let main_img_info = vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(main_img_view)
        .build();

    let ui_img_info = vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(ui_img_view)
        .build();

    let present_image_info = vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(swapchain_image.view)
        .build();

    unsafe {
        device.raw.cmd_bind_pipeline(
            cb.raw,
            vk::PipelineBindPoint::COMPUTE,
            present_shader.pipeline,
        );
    }

    todo!("replace push descriptors");

    /*unsafe {
        device.cmd_ext.push_descriptor.cmd_push_descriptor_set(
            cb.raw,
            vk::PipelineBindPoint::COMPUTE,
            present_shader.pipeline_layout,
            0,
            &[
                vk::WriteDescriptorSet::builder()
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&main_img_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&ui_img_info))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&present_image_info))
                    .build(),
            ],
        );

        let push_constants: [f32; 4] = [
            main_tex_extent[0] as f32,
            main_tex_extent[1] as f32,
            swapchain_extent[0] as f32,
            swapchain_extent[1] as f32,
        ];

        device.raw.cmd_push_constants(
            cb.raw,
            present_shader.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            crate::bytes::as_byte_slice(&push_constants),
        );
        device.raw.cmd_dispatch(
            cb.raw,
            (swapchain_extent[0] + 7) / 8,
            (swapchain_extent[1] + 7) / 8,
            1,
        );
    }*/

    record_image_barrier(
        device,
        cb.raw,
        ImageBarrier::new(
            swapchain_image.image,
            vk_sync::AccessType::ComputeShaderWrite,
            vk_sync::AccessType::Present,
            vk::ImageAspectFlags::COLOR,
        ),
    );
}
