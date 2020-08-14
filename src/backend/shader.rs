use super::device::{Device, SamplerDesc};
use ash::{version::DeviceV1_0, vk};
use spirv_reflect::types::ReflectResourceTypeFlags;

pub fn create_descriptor_set_layouts(
    device: &Device,
    module: &spirv_reflect::ShaderModule,
    set_flags: &[(usize, vk::DescriptorSetLayoutCreateFlags)],
) -> Vec<vk::DescriptorSetLayout> {
    let descriptor_sets = module.enumerate_descriptor_sets(None).unwrap();
    //println!("Shader descriptor sets: {:#?}", descriptor_sets);

    let stage_flags = vk::ShaderStageFlags::COMPUTE;

    descriptor_sets
        .into_iter()
        .enumerate()
        .map(|(set_index, set)| {
            let set = &set.value;

            let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
                Vec::with_capacity(set.binding_refs.len());

            for binding in set.binding_refs.iter() {
                let binding = &binding.value;

                assert_ne!(binding.resource_type, ReflectResourceTypeFlags::UNDEFINED);
                match binding.resource_type {
                    ReflectResourceTypeFlags::CONSTANT_BUFFER_VIEW => todo!(),
                    ReflectResourceTypeFlags::SHADER_RESOURCE_VIEW => {
                        bindings.push(
                            vk::DescriptorSetLayoutBinding::builder()
                                .binding(binding.binding)
                                .descriptor_count(binding.count)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .stage_flags(stage_flags)
                                .build(),
                        );
                    }
                    ReflectResourceTypeFlags::SAMPLER => {
                        let name_prefix = "sampler_";
                        if let Some(spec) = binding.name.strip_prefix(name_prefix) {
                            let parts: Vec<_> = spec.split('_').collect();
                            assert!(parts.len() == 3);
                            let texel_filter = match parts[0] {
                                "nearest" => vk::Filter::NEAREST,
                                "linear" => vk::Filter::LINEAR,
                                _ => panic!("{}", parts[0]),
                            };
                            let mipmap_mode = match parts[1] {
                                "nearest" => vk::SamplerMipmapMode::NEAREST,
                                "linear" => vk::SamplerMipmapMode::LINEAR,
                                _ => panic!("{}", parts[1]),
                            };
                            let address_modes = match parts[2] {
                                "repeat" => vk::SamplerAddressMode::REPEAT,
                                "mirroredrepeat" => vk::SamplerAddressMode::MIRRORED_REPEAT,
                                "clamp" => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                                "clamptoborder" => vk::SamplerAddressMode::CLAMP_TO_BORDER,
                                _ => panic!("{}", parts[2]),
                            };
                            bindings.push(
                                vk::DescriptorSetLayoutBinding::builder()
                                    .descriptor_count(1)
                                    .descriptor_type(vk::DescriptorType::SAMPLER)
                                    .stage_flags(stage_flags)
                                    .binding(binding.binding)
                                    .immutable_samplers(&[device.get_sampler(SamplerDesc {
                                        texel_filter,
                                        mipmap_mode,
                                        address_modes,
                                    })])
                                    .build(),
                            );
                        } else {
                            panic!("{}", binding.name);
                        }
                    }
                    ReflectResourceTypeFlags::UNORDERED_ACCESS_VIEW => {
                        bindings.push(
                            vk::DescriptorSetLayoutBinding::builder()
                                .binding(binding.binding)
                                .descriptor_count(binding.count)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .stage_flags(stage_flags)
                                .build(),
                        );
                    }
                    _ => unimplemented!(),
                }
            }

            let flags = set_flags
                .iter()
                .find(|item| item.0 == set_index)
                .map(|flags| flags.1)
                .unwrap_or_default();

            unsafe {
                device
                    .raw
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::builder()
                            .flags(flags)
                            .bindings(&bindings)
                            .build(),
                        None,
                    )
                    .unwrap()
            }
        })
        .collect()
}
