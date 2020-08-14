use super::device::{Device, SamplerDesc};
use ash::{version::DeviceV1_0, vk};
use byte_slice_cast::AsSliceOf as _;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};

pub fn create_present_descriptor_set_and_pipeline(
    device: &Device,
) -> (vk::PipelineLayout, vk::Pipeline) {
    use std::ffi::CString;

    let shader_entry_name = CString::new("main").unwrap();
    let shader_spv = include_bytes!("../final_blit.spv");
    let module = spirv_reflect::ShaderModule::load_u8_data(shader_spv).unwrap();

    let descriptor_set_layouts = super::shader::create_descriptor_set_layouts(
        device,
        &module,
        &[(0, vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)],
    );

    let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&[vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 2 * 4,
        }]);

    unsafe {
        let shader_module = device
            .raw
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder()
                    .code((&shader_spv[..]).as_slice_of::<u32>().unwrap()),
                None,
            )
            .unwrap();

        let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&shader_entry_name);

        let pipeline_layout = device
            .raw
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_create_info.build())
            .layout(pipeline_layout);

        let pipeline = device
            .raw
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .expect("pipeline")[0];

        (pipeline_layout, pipeline)
    }
}
