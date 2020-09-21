use super::{
    barrier::*,
    device::{CommandBuffer, Device},
    shader::{
        create_compute_pipeline, ComputePipeline, ComputePipelineDesc, DescriptorSetLayoutOpts,
    },
    swapchain::SwapchainImage,
};
use ash::{version::DeviceV1_0, vk};

pub fn create_present_compute_shader(device: &Device) -> ComputePipeline {
    create_compute_pipeline(
        device,
        &(&include_bytes!("../final_blit.spv")[..]).to_owned(),
        &ComputePipelineDesc::builder()
            .descriptor_set_opts(&[(
                0,
                DescriptorSetLayoutOpts::builder()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR),
            )])
            .push_constants_bytes(2 * 4)
            .build()
            .unwrap(),
    )
}

pub fn blit_image_to_swapchain(
    device: &Device,
    cb: &CommandBuffer,
    swapchain_image: &SwapchainImage,
    present_source_image_view: vk::ImageView,
    present_shader: &ComputePipeline,
) {
    record_image_barrier(
        &device.raw,
        cb.raw,
        ImageBarrier::new(
            swapchain_image.image,
            vk_sync::AccessType::Present,
            vk_sync::AccessType::ComputeShaderWrite,
            vk::ImageAspectFlags::COLOR,
        )
        .with_discard(true),
    );

    let source_image_info = vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(present_source_image_view)
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

    unsafe {
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
                    .image_info(std::slice::from_ref(&source_image_info))
                    .build(),
                /*vk::WriteDescriptorSet::builder()
                .dst_set(present_descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&[vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(gui_texture_view)
                    .build()])
                .build(),*/
                vk::WriteDescriptorSet::builder()
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&present_image_info))
                    .build(),
            ],
        );

        // TODO
        let output_size_pixels = (1280u32, 720u32); // TODO
        let push_constants: (f32, f32) = (
            1.0 / output_size_pixels.0 as f32,
            1.0 / output_size_pixels.1 as f32,
        );
        device.raw.cmd_push_constants(
            cb.raw,
            present_shader.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            std::slice::from_raw_parts(&push_constants.0 as *const f32 as *const u8, 2 * 4),
        );
        device.raw.cmd_dispatch(
            cb.raw,
            (output_size_pixels.0 + 7) / 8,
            (output_size_pixels.1 + 7) / 8,
            1,
        );
    }

    record_image_barrier(
        &device.raw,
        cb.raw,
        ImageBarrier::new(
            swapchain_image.image,
            vk_sync::AccessType::ComputeShaderWrite,
            vk_sync::AccessType::Present,
            vk::ImageAspectFlags::COLOR,
        ),
    );
}
