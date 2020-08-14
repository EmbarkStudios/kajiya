use super::{
    device::Device,
    shader::{create_compute_shader, ComputeShader, ComputeShaderDesc},
};
use ash::vk;

pub fn create_present_compute_shader(device: &Device) -> ComputeShader {
    create_compute_shader(
        device,
        ComputeShaderDesc {
            spv: include_bytes!("../final_blit.spv"),
            entry_name: "main",
            descriptor_set_layout_flags: &[(
                0,
                vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
            )],
            push_constants_bytes: 2 * 4,
        },
    )
}
