use std::collections::HashMap;

use kajiya_backend::{
    ash::{version::DeviceV1_0, vk},
    rspirv_reflect,
    vulkan::device,
    MAX_BINDLESS_DESCRIPTOR_COUNT,
};

lazy_static::lazy_static! {
    pub static ref BINDLESS_DESCRIPTOR_SET_LAYOUT: HashMap<u32, rspirv_reflect::DescriptorInfo> = [
        (0, rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_BUFFER,
            dimensionality: rspirv_reflect::DescriptorDimensionality::Single,
            name: Default::default(),
        }),
        (1, rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_BUFFER,
            dimensionality: rspirv_reflect::DescriptorDimensionality::Single,
            name: Default::default(),
        }),
        (2, rspirv_reflect::DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::SAMPLED_IMAGE,
            dimensionality: rspirv_reflect::DescriptorDimensionality::RuntimeArray,
            name: Default::default(),
        }),
    ]
    .iter()
    .cloned()
    .collect();
}

pub fn create_bindless_descriptor_set(device: &device::Device) -> vk::DescriptorSet {
    let raw_device = &device.raw;

    let set_binding_flags = [
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
    ];

    let mut binding_flags_create_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
        .binding_flags(&set_binding_flags)
        .build();

    let descriptor_set_layout = unsafe {
        raw_device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_count(MAX_BINDLESS_DESCRIPTOR_COUNT as _)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .build(),
                    ])
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags_create_info)
                    .build(),
                None,
            )
            .unwrap()
    };

    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: MAX_BINDLESS_DESCRIPTOR_COUNT as _,
        },
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&descriptor_sizes)
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .max_sets(1);

    let descriptor_pool = unsafe {
        raw_device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    };

    let variable_descriptor_count = MAX_BINDLESS_DESCRIPTOR_COUNT as _;
    let mut variable_descriptor_count_allocate_info =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
            .descriptor_counts(std::slice::from_ref(&variable_descriptor_count))
            .build();

    let set = unsafe {
        raw_device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                    .push_next(&mut variable_descriptor_count_allocate_info)
                    .build(),
            )
            .unwrap()[0]
    };

    set
}
