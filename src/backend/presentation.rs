use super::device::{Device, SamplerDesc};
use ash::{version::DeviceV1_0, vk};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

pub fn create_present_descriptor_set_and_pipeline(
    device: &Device,
) -> (vk::PipelineLayout, vk::Pipeline) {
    let descriptor_set_layout = unsafe {
        device
            .raw
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .binding(0)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .binding(1)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .binding(2)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .binding(3)
                            .immutable_samplers(&[device.get_sampler(SamplerDesc {
                                texel_filter: vk::Filter::LINEAR,
                                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                                address_modes: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                            })])
                            .build(),
                    ])
                    .build(),
                None,
            )
            .unwrap()
    };

    let (pipeline_layout, pipeline) =
        create_present_compute_pipeline(&device.raw, descriptor_set_layout);

    (pipeline_layout, pipeline)
}

fn create_present_compute_pipeline(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> (vk::PipelineLayout, vk::Pipeline) {
    use std::ffi::CString;
    use std::io::Cursor;

    let shader_entry_name = CString::new("main").unwrap();
    let mut shader_spv = Cursor::new(&include_bytes!("../final_blit.spv")[..]);
    let shader_code = ash::util::read_spv(&mut shader_spv).expect("Failed to read shader spv");

    let descriptor_set_layouts = [descriptor_set_layout];
    let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&[vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 2 * 4,
        }]);

    unsafe {
        let shader_module = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder().code(&shader_code),
                None,
            )
            .unwrap();

        let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&shader_entry_name);

        let pipeline_layout = device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_create_info.build())
            .layout(pipeline_layout);

        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .expect("pipeline")[0];

        (pipeline_layout, pipeline)
    }
}
