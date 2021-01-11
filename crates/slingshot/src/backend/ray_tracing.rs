use std::borrow::Cow;

use super::device::Device;
use anyhow::Result;
use ash::{extensions::khr, version::DeviceV1_0, version::DeviceV1_2, vk};
use gpu_allocator::{AllocationCreateDesc, MemoryLocation};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RayTracingShaderType {
    RayGen = 0,
    Miss = 1,
    IntersectionHit = 2,
    AnyHit = 3,
    ClosestHit = 4,
}
const MAX_RAY_TRACING_SHADER_TYPE: usize = 5;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RayTracingProgramType {
    RayGen = 0,
    Miss = 1,
    Hit = 2,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RayTracingGeometryType {
    Triangle = 0,
    BoundingBox = 1,
}

/*
// TODO: Should make this just use generic shader and program desc types
#[derive(Clone, Default, Debug)]
pub struct RayTracingShaderDesc {
    pub entry_point: String,
    pub shader_data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct RayTracingProgramDesc {
    pub program_type: RayTracingProgramType,
    pub shaders: [Option<RayTracingShaderDesc>; MAX_RAY_TRACING_SHADER_TYPE],
    pub signature: RenderShaderSignatureDesc,
}*/

#[derive(Clone, Copy, Debug)]
pub struct RayTracingGeometryPart {
    pub index_count: usize,
    pub index_offset: usize, // offset into the index buffer in bytes
    pub max_vertex: u32, // the highest index of a vertex that will be addressed by a build command using this structure
}

#[derive(Clone, Debug)]
pub struct RayTracingGeometryDesc {
    pub geometry_type: RayTracingGeometryType,
    pub vertex_buffer: vk::DeviceAddress,
    pub index_buffer: vk::DeviceAddress,
    pub vertex_format: vk::Format,
    pub vertex_stride: usize,
    pub parts: Vec<RayTracingGeometryPart>,
}

// TODO
type RenderResourceHandle = ();

#[derive(Clone)]
pub struct RayTracingTopAccelerationDesc<'a> {
    pub instances: Vec<&'a RayTracingAcceleration>,
}

#[derive(Clone, Debug)]
pub struct RayTracingBottomAccelerationDesc {
    pub geometries: Vec<RayTracingGeometryDesc>,
}

#[derive(Clone, Debug)]
pub struct RayTracingPipelineStateDesc {
    pub programs: Vec<RenderResourceHandle>,
}

#[derive(Clone, Debug)]
pub struct RayTracingShaderTableDesc {
    pub pipeline_state: RenderResourceHandle,
    pub raygen_entry_count: u32,
    pub hit_entry_count: u32,
    pub miss_entry_count: u32,
}

pub struct RayTracingAcceleration {
    raw: vk::AccelerationStructureKHR,
    buffer: super::buffer::Buffer,
}

impl Device {
    pub fn create_ray_tracing_bottom_acceleration(
        &self,
        desc: &RayTracingBottomAccelerationDesc,
    ) -> Result<RayTracingAcceleration> {
        //log::trace!("Creating ray tracing bottom acceleration: {:?}", desc);

        let geometries: Result<Vec<ash::vk::AccelerationStructureGeometryKHR>> = desc
            .geometries
            .iter()
            .map(
                |desc| -> Result<ash::vk::AccelerationStructureGeometryKHR> {
                    let part: RayTracingGeometryPart = desc.parts[0];

                    let geometry = ash::vk::AccelerationStructureGeometryKHR::builder()
                        .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(ash::vk::AccelerationStructureGeometryDataKHR {
                            triangles:
                                ash::vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                                    .vertex_data(ash::vk::DeviceOrHostAddressConstKHR {
                                        device_address: desc.vertex_buffer,
                                    })
                                    .vertex_stride(desc.vertex_stride as _)
                                    .max_vertex(part.max_vertex)
                                    .vertex_format(desc.vertex_format)
                                    .index_data(ash::vk::DeviceOrHostAddressConstKHR {
                                        device_address: desc.index_buffer,
                                    })
                                    .index_type(ash::vk::IndexType::UINT32) // TODO
                                    .build(),
                        })
                        .flags(ash::vk::GeometryFlagsKHR::OPAQUE)
                        .build();

                    Ok(geometry)
                },
            )
            .collect();
        let geometries = geometries?;

        let build_range_infos: Vec<ash::vk::AccelerationStructureBuildRangeInfoKHR> = desc
            .geometries
            .iter()
            .map(|desc| {
                ash::vk::AccelerationStructureBuildRangeInfoKHR::builder()
                    .primitive_count(desc.parts[0].index_count as u32 / 3)
                    .build()
            })
            .collect();

        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(geometries.as_slice())
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .build();

        let max_primitive_counts: Vec<_> = desc
            .geometries
            .iter()
            .map(|desc| desc.parts[0].index_count as u32 / 3)
            .collect();

        // Create bottom-level acceleration structure

        self.create_ray_tracing_acceleration(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            geometry_info,
            &build_range_infos,
            &max_primitive_counts,
        )
    }

    pub fn create_ray_tracing_top_acceleration(
        &self,
        desc: &RayTracingTopAccelerationDesc<'_>,
    ) -> Result<RayTracingAcceleration> {
        //log::trace!("Creating ray tracing top acceleration: {:?}", desc);

        // Create instance buffer

        let instances: Vec<GeometryInstance> = desc
            .instances
            .iter()
            .map(|desc| {
                let blas_address = unsafe {
                    self.acceleration_structure_ext
                        .get_acceleration_structure_device_address(
                            self.raw.handle(),
                            &ash::vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                                .acceleration_structure(desc.raw)
                                .build(),
                        )
                };

                let transform: [f32; 12] = [
                    1.0, 0.0, 0.0, -0.0, //
                    0.0, 1.0, 0.0, -0.0, //
                    0.0, 0.0, 1.0, -0.0, //
                ];

                GeometryInstance::new(
                    transform,
                    0, /* instance id */
                    0xff,
                    0,
                    ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                        | ash::vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE,
                    blas_address,
                )
            })
            .collect();

        let instance_buffer_size = std::mem::size_of::<GeometryInstance>() * instances.len();
        let instance_buffer = self
            .create_buffer(
                super::buffer::BufferDesc {
                    size: instance_buffer_size,
                    usage: ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    mapped: false,
                },
                unsafe {
                    Some(std::slice::from_raw_parts(
                        instances.as_ptr() as *const u8,
                        instance_buffer_size,
                    ))
                },
            )
            .expect("TLAS instance buffer");
        let instance_buffer_address = instance_buffer.device_address(self);

        let geometry = ash::vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(ash::vk::GeometryTypeKHR::INSTANCES)
            .geometry(ash::vk::AccelerationStructureGeometryDataKHR {
                instances: ash::vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(ash::vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_address,
                    })
                    .build(),
            })
            .build();

        let build_range_infos = vec![ash::vk::AccelerationStructureBuildRangeInfoKHR::builder()
            .primitive_count(instances.len() as _)
            .build()];

        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .build();

        let max_primitive_counts = [instances.len() as u32];

        // Create top-level acceleration structure

        self.create_ray_tracing_acceleration(
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            geometry_info,
            &build_range_infos,
            &max_primitive_counts,
        )
    }

    fn create_ray_tracing_acceleration(
        &self,
        ty: vk::AccelerationStructureTypeKHR,
        mut geometry_info: vk::AccelerationStructureBuildGeometryInfoKHR,
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
        max_primitive_counts: &[u32],
    ) -> Result<RayTracingAcceleration> {
        let memory_requirements = unsafe {
            self.acceleration_structure_ext
                .get_acceleration_structure_build_sizes(
                    self.raw.handle(),
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &geometry_info,
                    &max_primitive_counts,
                )
        };

        log::info!(
            "Acceleration structure size: {}, scratch size: {}",
            memory_requirements.acceleration_structure_size,
            memory_requirements.build_scratch_size
        );

        let accel_buffer = self
            .create_buffer(
                super::buffer::BufferDesc {
                    size: memory_requirements.acceleration_structure_size as _,
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    mapped: false,
                },
                None,
            )
            .expect("Acceleration structure buffer");

        let accel_info = ash::vk::AccelerationStructureCreateInfoKHR::builder()
            .ty(ty)
            .buffer(accel_buffer.raw)
            .size(memory_requirements.acceleration_structure_size as _)
            .build();

        unsafe {
            let accel_raw = self
                .acceleration_structure_ext
                .create_acceleration_structure(&accel_info, None)
                .expect("create_acceleration_structure");

            let scratch_buffer = self
                .create_buffer(
                    super::buffer::BufferDesc {
                        size: memory_requirements.build_scratch_size as _,
                        usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        mapped: false,
                    },
                    None,
                )
                .expect("Acceleration structure scratch buffer");

            geometry_info.dst_acceleration_structure = accel_raw;
            geometry_info.scratch_data = ash::vk::DeviceOrHostAddressKHR {
                device_address: self.raw.get_buffer_device_address(
                    &ash::vk::BufferDeviceAddressInfo::builder().buffer(scratch_buffer.raw),
                ),
            };

            self.with_setup_cb(|cb| {
                self.acceleration_structure_ext
                    .cmd_build_acceleration_structures(
                        cb,
                        std::slice::from_ref(&geometry_info),
                        std::slice::from_ref(&build_range_infos),
                    );

                self.raw.cmd_pipeline_barrier(
                    cb,
                    ash::vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    ash::vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    ash::vk::DependencyFlags::empty(),
                    &[ash::vk::MemoryBarrier::builder()
                        .src_access_mask(
                            ash::vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                                | ash::vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                        )
                        .dst_access_mask(
                            ash::vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                                | ash::vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                        )
                        .build()],
                    &[],
                    &[],
                );
            });

            Ok(RayTracingAcceleration {
                raw: accel_raw,
                buffer: accel_buffer,
            })
        }
    }

    /*fn create_ray_tracing_pipeline_state(
        &self,
        handle: RenderResourceHandle,
        desc: &RayTracingPipelineStateDesc,
        debug_name: Cow<'static, str>,
    ) -> Result<()> {
        log::info!(
            "Creating ray tracing pipeline state: {}, {:?}",
            debug_name,
            desc
        );

        let device = Arc::clone(&self.logical_device);
        let raw_device = device.device();

        let programs = desc
            .programs
            .iter()
            .map(|prog| self.storage.get_typed::<RenderRayTracingProgramVk>(*prog))
            .collect::<Result<Vec<_>>>()?;

        let mut combined_layouts: Vec<DescriptorSetLayout> = Vec::new();

        for program in desc.programs.iter() {
            let program = self
                .storage
                .get_typed::<RenderRayTracingProgramVk>(*program)?;
            for shader in program.shaders.iter().filter_map(|s| s.as_ref()) {
                merge_descriptor_set_layouts(&shader.set_layouts, &mut combined_layouts);
            }
        }

        // Make the set indices go in 0...N order
        combined_layouts.sort_by(|ref a, ref b| a.set_index.cmp(&b.set_index));

        // Create descriptor set layout objects
        // TODO
        /*for layout in &mut combined_layouts {
            // Add in global static sampler bindings
            for sampler_binding in &data.sampler_layouts {
                layout.bindings.push(*sampler_binding);
            }
        }*/

        let mut descriptor_layouts: Vec<ash::vk::DescriptorSetLayout> = vec![
                ash::vk::DescriptorSetLayout::null();
                MAX_SHADER_PARAMETERS * 2 + MAX_RAYGEN_SHADER_PARAMETERS * 2
            ];

        let mut pool_sizes = Vec::new();
        for index in 0..combined_layouts.len() {
            let mut combined_layout = &mut combined_layouts[index];
            let create_info = ash::vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&combined_layout.bindings)
                .build();

            combined_layout.layout = unsafe {
                raw_device
                    .create_descriptor_set_layout(&create_info, None)
                    .unwrap()
            };

            assert_eq!(
                descriptor_layouts[combined_layout.set_index as usize],
                ash::vk::DescriptorSetLayout::null()
            );
            descriptor_layouts[combined_layout.set_index as usize] = combined_layout.layout;
            for binding in &combined_layout.bindings {
                tally_descriptor_pool_sizes(&mut pool_sizes, binding.descriptor_type);
            }
        }

        for descriptor_layout in descriptor_layouts[0..MAX_SHADER_PARAMETERS].iter_mut() {
            if *descriptor_layout == ash::vk::DescriptorSetLayout::null() {
                *descriptor_layout = self.empty_descriptor_set_layout;
            }
        }
        for (idx, descriptor_layout) in descriptor_layouts
            [MAX_SHADER_PARAMETERS..2 * MAX_SHADER_PARAMETERS]
            .iter_mut()
            .enumerate()
        {
            *descriptor_layout = self.cbuffer_descriptor_set_layouts[idx];
        }

        let raygen_descriptor_set_idx = 2 * MAX_SHADER_PARAMETERS;

        // TODO
        descriptor_layouts[raygen_descriptor_set_idx] = self.empty_descriptor_set_layout;
        descriptor_layouts[raygen_descriptor_set_idx + 1] = self.cbuffer_descriptor_set_layouts[0];

        /*let descriptor_set_layout = unsafe {
            raw_device
                .create_descriptor_set_layout(
                    &ash::vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&[
                            ash::vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(
                                    ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                                )
                                .stage_flags(ash::vk::ShaderStageFlags::RAYGEN_KHR)
                                .binding(0)
                                .build(),
                            ash::vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(ash::vk::DescriptorType::STORAGE_IMAGE)
                                .stage_flags(ash::vk::ShaderStageFlags::RAYGEN_KHR)
                                .binding(1)
                                .build(),
                            /*ash::vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(3)
                            .descriptor_type(ash::vk::DescriptorType::UNIFORM_BUFFER)
                            .stage_flags(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                            .binding(2)
                            .build(),*/
                        ])
                        .build(),
                    None,
                )
                .unwrap()
        };*/

        let create_shader_module =
            |desc: &RayTracingShaderDesc| -> Result<(ash::vk::ShaderModule, String)> {
                let shader_data: *const u8 = desc.shader_data.as_ptr();
                let shader_info = ash::vk::ShaderModuleCreateInfo {
                    flags: Default::default(),
                    code_size: desc.shader_data.len(), // in bytes
                    p_code: shader_data as *const u32,
                    ..Default::default()
                };

                let module = unsafe { raw_device.create_shader_module(&shader_info, None) }
                    .map_err(|err| {
                        Error::backend(format!("failed to create shader - {:?}", err))
                    })?;
                let entry_point = desc.entry_point.clone();

                Ok((module, entry_point))
            };

        let mut shader_groups = vec![ash::vk::RayTracingShaderGroupCreateInfoKHR::default(); 2];
        let mut shader_stages = vec![ash::vk::PipelineShaderStageCreateInfo::default(); 2];

        // Keep entry point names alive, since build() forgets references.
        let mut entry_points: Vec<std::ffi::CString> = Vec::new();

        let mut raygen_found = false;
        let mut miss_found = false;

        const RAYGEN_IDX: usize = 0;
        const MISS_IDX: usize = 1;

        for program in programs {
            let program = &*program;
            match program.program_type {
                RayTracingProgramType::RayGen => {
                    assert!(!raygen_found, "only one raygen shader supported right now");
                    raygen_found = true;

                    let (module, entry_point) = create_shader_module(
                        &program.shaders[RayTracingShaderType::RayGen as usize]
                            .as_ref()
                            .expect("raygen shader")
                            .desc,
                    )?;

                    entry_points.push(std::ffi::CString::new(entry_point).unwrap());
                    let entry_point = &**entry_points.last().unwrap();

                    let stage = ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::RAYGEN_KHR)
                        .module(module)
                        .name(entry_point)
                        .build();

                    let group = ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                        .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                        .general_shader(RAYGEN_IDX as _)
                        .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .intersection_shader(ash::vk::SHADER_UNUSED_KHR)
                        .build();

                    shader_stages[RAYGEN_IDX] = stage;
                    shader_groups[RAYGEN_IDX] = group;
                }
                RayTracingProgramType::Miss => {
                    assert!(!miss_found, "only one miss shader supported right now");
                    miss_found = true;

                    let (module, entry_point) = create_shader_module(
                        &program.shaders[RayTracingShaderType::Miss as usize]
                            .as_ref()
                            .expect("miss shader")
                            .desc,
                    )?;

                    entry_points.push(std::ffi::CString::new(entry_point).unwrap());
                    let entry_point = &**entry_points.last().unwrap();

                    let stage = ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::MISS_KHR)
                        .module(module)
                        .name(entry_point)
                        .build();

                    let group = ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                        .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                        .general_shader(MISS_IDX as _)
                        .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .intersection_shader(ash::vk::SHADER_UNUSED_KHR)
                        .build();

                    shader_stages[MISS_IDX] = stage;
                    shader_groups[MISS_IDX] = group;
                }
                RayTracingProgramType::Hit => {
                    // TODO: procedural geometry

                    let (module, entry_point) = create_shader_module(
                        &program.shaders[RayTracingShaderType::ClosestHit as usize]
                            .as_ref()
                            .expect("closest hit shader")
                            .desc,
                    )?;

                    entry_points.push(std::ffi::CString::new(entry_point).unwrap());
                    let entry_point = &**entry_points.last().unwrap();

                    let stage = ash::vk::PipelineShaderStageCreateInfo::builder()
                        .stage(ash::vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                        .module(module)
                        .name(entry_point)
                        .build();

                    let group = ash::vk::RayTracingShaderGroupCreateInfoKHR::builder()
                        .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                        .general_shader(ash::vk::SHADER_UNUSED_KHR)
                        .closest_hit_shader(shader_stages.len() as _)
                        .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .intersection_shader(ash::vk::SHADER_UNUSED_KHR)
                        .build();

                    shader_stages.push(stage);
                    shader_groups.push(group);
                }
            }
        }

        assert!(raygen_found);
        assert!(miss_found);
        assert!(
            shader_groups.len() >= 3,
            "Must supply at least closest hit shader"
        );

        let layout_create_info =
            ash::vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_layouts);
        let pipeline_layout = unsafe {
            raw_device
                .create_pipeline_layout(&layout_create_info, None)
                .unwrap()
        };

        let pipeline = unsafe {
            self.ray_tracing
                .create_ray_tracing_pipelines(
                    self.pipeline_cache,
                    &[ash::vk::RayTracingPipelineCreateInfoKHR::builder()
                        .stages(&shader_stages)
                        .groups(&shader_groups)
                        .max_recursion_depth(1) // TODO
                        .layout(pipeline_layout)
                        .build()],
                    None,
                )
                .unwrap()[0]
        };

        let resource: Arc<RwLock<Box<dyn RenderResourceBase>>> =
            Arc::new(RwLock::new(Box::new(RenderRayTracingPipelineStateVk {
                name: debug_name.to_string().into(),
                pipeline,
                data: RenderPipelineLayoutVk {
                    static_samplers: Vec::new(), // TODO
                    pipeline_layout,
                    combined_layouts,
                    sampler_layouts: Vec::new(), // TODO
                    pool_sizes,
                },
                descriptor_set_layouts: descriptor_layouts,
            })));

        self.storage.put(handle, resource)?;
        Ok(())
    }

    fn create_ray_tracing_shader_table(
        &self,
        handle: RenderResourceHandle,
        desc: &RayTracingShaderTableDesc,
        debug_name: Cow<'static, str>,
    ) -> Result<()> {
        log::info!(
            "Creating ray tracing shader table: {}, {:?}",
            debug_name,
            desc
        );

        let shader_group_handle_size =
            self.ray_tracing_properties.shader_group_handle_size as usize;
        let group_count =
            (desc.raygen_entry_count + desc.miss_entry_count + desc.hit_entry_count) as usize;
        let group_handles_size = (shader_group_handle_size * group_count) as usize;

        let pipeline = &*self
            .storage
            .get_typed::<RenderRayTracingPipelineStateVk>(desc.pipeline_state)?;

        let mut group_handles: Vec<u8> = vec![0u8; group_handles_size];
        unsafe {
            self.ray_tracing
                .get_ray_tracing_shader_group_handles(
                    pipeline.pipeline,
                    0,
                    group_count as _,
                    &mut group_handles,
                )
                .unwrap();
        }

        let prog_size = shader_group_handle_size;

        let create_binding_table = |entry_offset: u32, entry_count: u32| {
            let mut shader_binding_table_data = vec![0u8; (entry_count as usize * prog_size) as _];

            for dst in 0..(entry_count as usize) {
                let src = dst + entry_offset as usize;
                shader_binding_table_data
                    [dst * prog_size..dst * prog_size + shader_group_handle_size]
                    .copy_from_slice(
                        &group_handles[src * shader_group_handle_size
                            ..src * shader_group_handle_size + shader_group_handle_size],
                    );
            }

            let mut shader_binding_table = BufferResource::new(
                shader_binding_table_data.len() as _,
                ash::vk::BufferUsageFlags::TRANSFER_SRC,
                ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
                self.logical_device.clone(),
                self.global_allocator.clone(),
            );
            shader_binding_table.store(&shader_binding_table_data);
            shader_binding_table
        };

        let raygen_shader_binding_table = create_binding_table(0, desc.raygen_entry_count);
        let miss_shader_binding_table =
            create_binding_table(desc.raygen_entry_count, desc.miss_entry_count);
        let hit_shader_binding_table = create_binding_table(
            desc.raygen_entry_count + desc.miss_entry_count,
            desc.hit_entry_count,
        );

        //dbg!(shader_group_handle_size);
        //dbg!(self.ray_tracing_properties.shader_group_base_alignment);

        /*let descriptor_sets = unsafe {
            let descriptor_pool_info = ash::vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pipeline.data.pool_sizes)
                .max_sets(pipeline.descriptor_set_layouts.len() as _);

            let device = Arc::clone(&self.logical_device);
            let raw_device = device.device();

            let descriptor_pool = raw_device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();

            raw_device
                .allocate_descriptor_sets(
                    &ash::vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&pipeline.descriptor_set_layouts)
                        .build(),
                )
                .unwrap_or_else(|_| {
                    panic!(
                        "Failed to allocate pipeline {:?} from pool with sizes {:?}",
                        pipeline.data.combined_layouts, pipeline.data.pool_sizes
                    )
                })
        };*/

        let resource: Arc<RwLock<Box<dyn RenderResourceBase>>> =
            Arc::new(RwLock::new(Box::new(RenderRayTracingShaderTableVk {
                name: debug_name.to_string().into(),
                raygen_shader_binding_table: ash::vk::StridedBufferRegionKHR {
                    buffer: raygen_shader_binding_table.buffer,
                    offset: 0,
                    stride: prog_size as u64,
                    size: (prog_size * desc.raygen_entry_count as usize) as u64,
                },
                raygen_shader_binding_table_buffer: Some(raygen_shader_binding_table),
                miss_shader_binding_table: ash::vk::StridedBufferRegionKHR {
                    buffer: miss_shader_binding_table.buffer,
                    offset: 0,
                    stride: prog_size as u64,
                    size: (prog_size * desc.miss_entry_count as usize) as u64,
                },
                miss_shader_binding_table_buffer: Some(miss_shader_binding_table),
                hit_shader_binding_table: ash::vk::StridedBufferRegionKHR {
                    buffer: hit_shader_binding_table.buffer,
                    offset: 0,
                    stride: prog_size as u64,
                    size: (prog_size * desc.hit_entry_count as usize) as u64,
                },
                hit_shader_binding_table_buffer: Some(hit_shader_binding_table),
                callable_shader_binding_table_buffer: None,
                callable_shader_binding_table: ash::vk::StridedBufferRegionKHR {
                    buffer: Default::default(),
                    offset: 0,
                    stride: 0,
                    size: 0,
                },
                descriptor_sets: Default::default(),
                cbuffer_descriptor_sets: Default::default(),
                cbuffer_dynamic_offsets: vec![
                    0u32;
                    MAX_SHADER_PARAMETERS + MAX_RAYGEN_SHADER_PARAMETERS
                ],
            })));

        self.storage.put(handle, resource)?;
        Ok(())
    }

    // Ray tracing features are only supported on some devices
    fn create_ray_tracing_program(
        &self,
        handle: RenderResourceHandle,
        desc: &RayTracingProgramDesc,
        debug_name: Cow<'static, str>,
    ) -> Result<()> {
        log::info!("Creating ray tracing program: {}, {:?}", debug_name, desc);

        // Tom-NOTE: can't really do anything here because the shader modules
        // need to have their layouts merged in the context of a ray-tracing pipeline.
        // This just persists the desc for use in `create_ray_tracing_pipeline_state`

        let device = Arc::clone(&self.logical_device);
        let raw_device = device.device();

        let mut remapped_shaders: [Option<RenderRayTracingShaderVk>; MAX_RAY_TRACING_SHADER_TYPE] =
            [None, None, None, None, None];

        for (shader_idx, desc) in desc.shaders.iter().enumerate() {
            if let Some(desc) = desc {
                let mut remaps: Vec<BindingSetRemap> = Vec::new();
                match spirv_reflect::ShaderModule::load_u8_data(&desc.shader_data) {
                    Ok(mut reflect_module) => {
                        let mut srv_count = 0;
                        let mut smp_count = 0;
                        let mut uav_count = 0;

                        let descriptor_sets =
                            reflect_module.enumerate_descriptor_sets(None).unwrap();
                        log::trace!("Shader descriptor sets: {:#?}", descriptor_sets);

                        for set_index in 0..descriptor_sets.len() {
                            let set = &descriptor_sets[set_index].value;

                            for (binding_index, binding) in set.binding_refs.iter().enumerate() {
                                let binding = &binding.value;

                                assert_ne!(
                                    binding.resource_type,
                                    ReflectResourceTypeFlags::UNDEFINED
                                );

                                match binding.resource_type {
                                    ReflectResourceTypeFlags::CONSTANT_BUFFER_VIEW => {
                                        assert_eq!(binding.binding, 0);
                                        remaps.push(BindingSetRemap {
                                            binding_index: binding_index as u32,
                                            old_binding: binding.binding,
                                            new_binding: binding.binding,
                                            old_set: set_index as u32,
                                            new_set: SET_OFFSET + set_index as u32,
                                        });
                                        // Make sure we don't use more spaces/sets than MAX_SHADER_PARAMETERS
                                        assert!(
                                            SET_OFFSET as usize + set_index
                                                >= descriptor_sets.len()
                                        );
                                    }
                                    ReflectResourceTypeFlags::SHADER_RESOURCE_VIEW => {
                                        remaps.push(BindingSetRemap {
                                            binding_index: binding_index as u32,
                                            old_binding: binding.binding,
                                            new_binding: srv_count + SRV_OFFSET,
                                            old_set: set_index as u32,
                                            new_set: ARG_OFFSET + set_index as u32,
                                        });
                                        srv_count += 1;
                                    }
                                    ReflectResourceTypeFlags::SAMPLER => {
                                        remaps.push(BindingSetRemap {
                                            binding_index: binding_index as u32,
                                            old_binding: binding.binding,
                                            new_binding: smp_count + SMP_OFFSET,
                                            old_set: set_index as u32,
                                            new_set: ARG_OFFSET + set_index as u32,
                                        });
                                        smp_count += 1;
                                    }
                                    ReflectResourceTypeFlags::UNORDERED_ACCESS_VIEW => {
                                        remaps.push(BindingSetRemap {
                                            binding_index: binding_index as u32,
                                            old_binding: binding.binding,
                                            new_binding: uav_count + UAV_OFFSET,
                                            old_set: set_index as u32,
                                            new_set: ARG_OFFSET + set_index as u32,
                                        });
                                        uav_count += 1;
                                    }
                                    ReflectResourceTypeFlags::ACCELERATION_STRUCTURE => {
                                        remaps.push(BindingSetRemap {
                                            binding_index: binding_index as u32,
                                            old_binding: binding.binding,
                                            new_binding: srv_count + SRV_OFFSET,
                                            old_set: set_index as u32,
                                            new_set: ARG_OFFSET + set_index as u32,
                                        });
                                        srv_count += 1;
                                    }
                                    _ => unimplemented!(),
                                }
                            }

                            for remap in &remaps {
                                let binding = &set.binding_refs[remap.binding_index as usize];
                                match reflect_module.change_descriptor_binding_numbers(
                                    binding,
                                    Some(remap.new_binding),
                                    Some(remap.new_set),
                                ) {
                                    Ok(_) => {}
                                    Err(err) => {
                                        return Err(Error::backend(format!(
                                            "failed to patch descriptor binding - {:?}",
                                            err
                                        )));
                                    }
                                }
                            }
                        }

                        // Create descriptor set layouts
                        let descriptor_sets =
                            reflect_module.enumerate_descriptor_sets(None).unwrap();
                        let mut set_layouts: Vec<(
                            u32, /* set index */
                            Vec<ash::vk::DescriptorSetLayoutBinding>,
                        )> = Vec::with_capacity(descriptor_sets.len());
                        for set_index in 0..descriptor_sets.len() {
                            let reflected_set = &descriptor_sets[set_index].value;

                            let mut layout_bindings: Vec<ash::vk::DescriptorSetLayoutBinding> =
                                Vec::with_capacity(reflected_set.binding_refs.len());

                            for (binding_index, reflected_binding) in
                                reflected_set.binding_refs.iter().enumerate()
                            {
                                let reflected_binding = &reflected_binding.value;

                                let mut layout_binding =
                                    ash::vk::DescriptorSetLayoutBinding::default();
                                layout_binding.binding = reflected_binding.binding;
                                layout_binding.descriptor_type =
                                    reflection_descriptor_to_vk(reflected_binding.descriptor_type);

                                layout_binding.descriptor_count = 1;
                                for dim in &reflected_binding.array.dims {
                                    layout_binding.descriptor_count *= dim;
                                }

                                layout_binding.stage_flags = reflection_shader_stage_to_vk(
                                    reflect_module.get_shader_stage(),
                                );

                                layout_bindings.push(layout_binding);
                            }
                            set_layouts.push((reflected_set.set, layout_bindings));
                        }

                        let patched_spv = reflect_module.get_code();

                        remapped_shaders[shader_idx] = Some(RenderRayTracingShaderVk {
                            desc: RayTracingShaderDesc {
                                entry_point: desc.entry_point.clone(),
                                shader_data: render_core::utilities::typed_to_bytes(patched_spv)
                                    .to_owned(),
                            },
                            set_layouts,
                        });
                    }
                    Err(err) => {
                        return Err(Error::backend(format!(
                            "failed to parse shader - {:#?}",
                            err
                        )))
                    }
                }
            }
        }

        let resource: Arc<RwLock<Box<dyn RenderResourceBase>>> =
            Arc::new(RwLock::new(Box::new(RenderRayTracingProgramVk {
                name: debug_name.to_string().into(),
                program_type: desc.program_type,
                shaders: remapped_shaders,
            })));

        self.storage.put(handle, resource)?;
        Ok(())
    }*/
}

/*#[derive(Clone, Debug)]
pub struct RenderRayTracingPipelineStateVk {
    pub name: Cow<'static, str>,
    pub pipeline: ash::vk::Pipeline,
    pub data: RenderPipelineLayoutVk,
    pub descriptor_set_layouts: Vec<ash::vk::DescriptorSetLayout>, // TODO: nuke
}

#[derive(Clone, Debug)]
pub struct RenderRayTracingShaderVk {
    pub desc: RayTracingShaderDesc,
    pub set_layouts: Vec<(
        u32, /* set index */
        Vec<ash::vk::DescriptorSetLayoutBinding>,
    )>,
}

#[derive(Clone, Debug)]
pub struct RenderRayTracingProgramVk {
    pub name: Cow<'static, str>,
    pub program_type: RayTracingProgramType,
    pub shaders: [Option<RenderRayTracingShaderVk>; MAX_RAY_TRACING_SHADER_TYPE],
}*/

#[derive(Clone, Debug)]
pub struct RenderRayTracingGeometryVk {
    pub name: Cow<'static, str>,
}

/*#[derive(Clone, Debug)]
pub struct RenderRayTracingBottomAccelerationVk {
    pub name: Cow<'static, str>,
    pub allocation: vk_mem::Allocation,
    pub handle: ash::vk::AccelerationStructureKHR,
}

#[derive(Clone, Debug)]
pub struct RenderRayTracingTopAccelerationVk {
    pub name: Cow<'static, str>,
    pub allocation: vk_mem::Allocation,
    pub handle: ash::vk::AccelerationStructureKHR,
}*/

/*#[derive(Clone, Debug)]
pub struct RenderRayTracingShaderTableVk {
    pub name: Cow<'static, str>,
    pub raygen_shader_binding_table_buffer: Option<crate::device::BufferResource>,
    pub raygen_shader_binding_table: ash::vk::StridedBufferRegionKHR,
    pub miss_shader_binding_table_buffer: Option<crate::device::BufferResource>,
    pub miss_shader_binding_table: ash::vk::StridedBufferRegionKHR,
    pub hit_shader_binding_table_buffer: Option<crate::device::BufferResource>,
    pub hit_shader_binding_table: ash::vk::StridedBufferRegionKHR,
    pub callable_shader_binding_table_buffer: Option<crate::device::BufferResource>,
    pub callable_shader_binding_table: ash::vk::StridedBufferRegionKHR,
    pub descriptor_sets: Vec<ash::vk::DescriptorSet>,
    pub cbuffer_descriptor_sets: Vec<ash::vk::DescriptorSet>,
    pub cbuffer_dynamic_offsets: Vec<u32>,
}
*/

#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct GeometryInstance {
    transform: [f32; 12],
    instance_id_and_mask: u32,
    instance_sbt_offset_and_flags: u32,
    blas_address: vk::DeviceAddress,
}

impl GeometryInstance {
    fn new(
        transform: [f32; 12],
        id: u32,
        mask: u8,
        sbt_offset: u32,
        flags: ash::vk::GeometryInstanceFlagsKHR,
        blas_address: vk::DeviceAddress,
    ) -> Self {
        let mut instance = GeometryInstance {
            transform,
            instance_id_and_mask: 0,
            instance_sbt_offset_and_flags: 0,
            blas_address,
        };
        instance.set_id(id);
        instance.set_mask(mask);
        instance.set_sbt_offset(sbt_offset);
        instance.set_flags(flags);
        instance
    }

    fn set_id(&mut self, id: u32) {
        let id = id & 0x00ffffff;
        self.instance_id_and_mask |= id;
    }

    fn set_mask(&mut self, mask: u8) {
        let mask = mask as u32;
        self.instance_id_and_mask |= mask << 24;
    }

    fn set_sbt_offset(&mut self, offset: u32) {
        let offset = offset & 0x00ffffff;
        self.instance_sbt_offset_and_flags |= offset;
    }

    fn set_flags(&mut self, flags: ash::vk::GeometryInstanceFlagsKHR) {
        let flags = flags.as_raw() as u32;
        self.instance_sbt_offset_and_flags |= flags << 24;
    }
}
