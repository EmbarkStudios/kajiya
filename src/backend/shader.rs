use super::device::{Device, SamplerDesc};
use ash::{version::DeviceV1_0, vk};
use byte_slice_cast::AsSliceOf as _;
use derive_builder::Builder;
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
                        if let Some(mut spec) = binding.name.strip_prefix(name_prefix) {
                            let texel_filter = match &spec[..1] {
                                "n" => vk::Filter::NEAREST,
                                "l" => vk::Filter::LINEAR,
                                _ => panic!("{}", &spec[..1]),
                            };
                            spec = &spec[1..];

                            let mipmap_mode = match &spec[..1] {
                                "n" => vk::SamplerMipmapMode::NEAREST,
                                "l" => vk::SamplerMipmapMode::LINEAR,
                                _ => panic!("{}", &spec[..1]),
                            };
                            spec = &spec[1..];

                            let address_modes = match spec {
                                "r" => vk::SamplerAddressMode::REPEAT,
                                "mr" => vk::SamplerAddressMode::MIRRORED_REPEAT,
                                "c" => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                                "cb" => vk::SamplerAddressMode::CLAMP_TO_BORDER,
                                _ => panic!("{}", spec),
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

pub struct ComputeShader {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct ComputeShaderDesc<'a, 'b> {
    pub spirv: &'a [u8],
    pub entry_name: &'b str,
    #[builder(setter(strip_option), default)]
    pub descriptor_set_layout_flags: Option<&'a [(usize, vk::DescriptorSetLayoutCreateFlags)]>,
    #[builder(default)]
    pub push_constants_bytes: usize,
}

impl<'a, 'b> ComputeShaderDesc<'a, 'b> {
    pub fn builder() -> ComputeShaderDescBuilder<'a, 'b> {
        ComputeShaderDescBuilder::default()
    }
}

pub fn create_compute_shader(device: &Device, desc: ComputeShaderDesc) -> ComputeShader {
    use std::ffi::CString;

    let shader_entry_name = CString::new(desc.entry_name).unwrap();
    let shader_spv = desc.spirv;
    let module = spirv_reflect::ShaderModule::load_u8_data(shader_spv).unwrap();

    let descriptor_set_layouts = super::shader::create_descriptor_set_layouts(
        device,
        &module,
        desc.descriptor_set_layout_flags.unwrap_or(&[]),
    );

    let mut layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

    let push_constant_ranges = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: desc.push_constants_bytes as _,
    };

    if desc.push_constants_bytes > 0 {
        layout_create_info =
            layout_create_info.push_constant_ranges(std::slice::from_ref(&push_constant_ranges));
    }

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
            // TODO: pipeline cache
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .expect("pipeline")[0];

        ComputeShader {
            pipeline_layout,
            pipeline,
        }
    }
}
