use std::{borrow::Cow, ffi::CString};

use crate::chunky_list::TempList;

use super::{
    device::Device,
    shader::{
        merge_shader_stage_layouts, PipelineShader, ShaderPipelineCommon, ShaderPipelineStage,
    },
};
use anyhow::Result;
use ash::{extensions::khr, version::DeviceV1_0, version::DeviceV1_2, vk};
use byte_slice_cast::AsSliceOf;
use gpu_allocator::{AllocationCreateDesc, MemoryLocation};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RayTracingGeometryType {
    Triangle = 0,
    BoundingBox = 1,
}

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
    pub raygen_entry_count: u32,
    pub hit_entry_count: u32,
    pub miss_entry_count: u32,
}

pub struct RayTracingAcceleration {
    pub(crate) raw: vk::AccelerationStructureKHR,
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
                    /*ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                    | */
                    ash::vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE,
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

    fn create_ray_tracing_shader_table(
        &self,
        desc: &RayTracingShaderTableDesc,
        pipeline: vk::Pipeline,
    ) -> Result<RayTracingShaderTable> {
        log::trace!("Creating ray tracing shader table: {:?}", desc);

        let shader_group_handle_size = self
            .ray_tracing_pipeline_properties
            .shader_group_handle_size as usize;
        let group_count =
            (desc.raygen_entry_count + desc.miss_entry_count + desc.hit_entry_count) as usize;
        let group_handles_size = (shader_group_handle_size * group_count) as usize;

        let group_handles: Vec<u8> = unsafe {
            self.ray_tracing_pipeline_ext
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count as _,
                    group_handles_size,
                )
                .unwrap()
        };

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

            self.create_buffer(
                super::buffer::BufferDesc {
                    size: shader_binding_table_data.len(),
                    usage: vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    mapped: false,
                },
                Some(&shader_binding_table_data),
            )
            .expect("SBT sub-buffer")
        };

        let raygen_shader_binding_table = create_binding_table(0, desc.raygen_entry_count);
        let miss_shader_binding_table =
            create_binding_table(desc.raygen_entry_count, desc.miss_entry_count);
        let hit_shader_binding_table = create_binding_table(
            desc.raygen_entry_count + desc.miss_entry_count,
            desc.hit_entry_count,
        );

        Ok(RayTracingShaderTable {
            raygen_shader_binding_table: vk::StridedDeviceAddressRegionKHR {
                device_address: raygen_shader_binding_table.device_address(self),
                stride: prog_size as u64,
                size: (prog_size * desc.raygen_entry_count as usize) as u64,
            },
            raygen_shader_binding_table_buffer: Some(raygen_shader_binding_table),
            miss_shader_binding_table: vk::StridedDeviceAddressRegionKHR {
                device_address: miss_shader_binding_table.device_address(self),
                stride: prog_size as u64,
                size: (prog_size * desc.miss_entry_count as usize) as u64,
            },
            miss_shader_binding_table_buffer: Some(miss_shader_binding_table),
            hit_shader_binding_table: vk::StridedDeviceAddressRegionKHR {
                device_address: hit_shader_binding_table.device_address(self),
                stride: prog_size as u64,
                size: (prog_size * desc.hit_entry_count as usize) as u64,
            },
            hit_shader_binding_table_buffer: Some(hit_shader_binding_table),
            callable_shader_binding_table_buffer: None,
            callable_shader_binding_table: vk::StridedDeviceAddressRegionKHR {
                device_address: Default::default(),
                stride: 0,
                size: 0,
            },
        })
    }
}

pub struct RayTracingPipeline {
    pub common: ShaderPipelineCommon,
    pub sbt: RayTracingShaderTable,
}

impl std::ops::Deref for RayTracingPipeline {
    type Target = ShaderPipelineCommon;

    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

pub fn create_ray_tracing_pipeline(
    device: &Device,
    shaders: &[PipelineShader<&[u8]>],
) -> anyhow::Result<RayTracingPipeline> {
    let stage_layouts = shaders
        .iter()
        .map(|desc| {
            rspirv_reflect::Reflection::new_from_spirv(desc.code)
                .unwrap_or_else(|err| panic!("Failed compiling shader {:?}:\n{:?}", desc.desc, err))
                .get_descriptor_sets()
                .unwrap()
        })
        .collect::<Vec<_>>();

    log::info!("{:#?}", stage_layouts);

    let (descriptor_set_layouts, set_layout_info) = super::shader::create_descriptor_set_layouts(
        device,
        &merge_shader_stage_layouts(stage_layouts),
        vk::ShaderStageFlags::RAYGEN_KHR,
        //desc.descriptor_set_layout_flags.unwrap_or(&[]),  // TODO: merge flags
        &Default::default(),
    );

    unsafe {
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .build();

        let pipeline_layout = device
            .raw
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let mut shader_groups = vec![ash::vk::RayTracingShaderGroupCreateInfoKHR::default(); 2];
        let mut shader_stages = vec![ash::vk::PipelineShaderStageCreateInfo::default(); 2];

        // Keep entry point names alive, since build() forgets references.
        let mut entry_points: Vec<std::ffi::CString> = Vec::new();

        let mut raygen_found = false;
        let mut miss_found = false;

        const RAYGEN_IDX: usize = 0;
        const MISS_IDX: usize = 1;

        let create_shader_module =
            |desc: &PipelineShader<&[u8]>| -> (ash::vk::ShaderModule, String) {
                let shader_info = vk::ShaderModuleCreateInfo::builder()
                    .code(desc.code.as_slice_of::<u32>().unwrap());

                let shader_module = device
                    .raw
                    .create_shader_module(&shader_info, None)
                    .expect("Shader module error");

                (shader_module, desc.desc.entry_name.clone())
            };

        for desc in shaders {
            match desc.desc.stage {
                ShaderPipelineStage::RayGen => {
                    assert!(!raygen_found, "only one raygen shader supported right now");
                    raygen_found = true;

                    let (module, entry_point) = create_shader_module(desc);

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
                ShaderPipelineStage::RayMiss => {
                    assert!(!miss_found, "only one miss shader supported right now");
                    miss_found = true;

                    let (module, entry_point) = create_shader_module(desc);

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
                ShaderPipelineStage::RayClosestHit => {
                    // TODO: procedural geometry

                    let (module, entry_point) = create_shader_module(desc);

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
                _ => unimplemented!(),
            }
        }

        assert!(raygen_found);
        assert!(miss_found);
        assert!(
            shader_groups.len() >= 3,
            "Must supply at least closest hit shader"
        );

        let pipeline = device
            .ray_tracing_pipeline_ext
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[ash::vk::RayTracingPipelineCreateInfoKHR::builder()
                    .stages(&shader_stages)
                    .groups(&shader_groups)
                    .max_pipeline_ray_recursion_depth(1) // TODO
                    .layout(pipeline_layout)
                    .build()],
                None,
            )
            .unwrap()[0];

        let mut descriptor_pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for bindings in set_layout_info.iter() {
            for ty in bindings.values() {
                if let Some(mut dps) = descriptor_pool_sizes.iter_mut().find(|item| item.ty == *ty)
                {
                    dps.descriptor_count += 1;
                } else {
                    descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                        ty: *ty,
                        descriptor_count: 1,
                    })
                }
            }
        }

        let sbt = device
            .create_ray_tracing_shader_table(
                &RayTracingShaderTableDesc {
                    // TODO
                    raygen_entry_count: 1,
                    hit_entry_count: 1,
                    miss_entry_count: 1,
                },
                pipeline,
            )
            .expect("SBT");

        Ok(RayTracingPipeline {
            common: ShaderPipelineCommon {
                pipeline_layout,
                pipeline,
                //render_pass: desc.render_pass.clone(),
                set_layout_info,
                descriptor_pool_sizes,
                descriptor_set_layouts,
                pipeline_bind_point: vk::PipelineBindPoint::RAY_TRACING_KHR,
            },
            sbt,
        })
    }
}

pub struct RayTracingShaderTable {
    pub raygen_shader_binding_table_buffer: Option<super::buffer::Buffer>,
    pub raygen_shader_binding_table: vk::StridedDeviceAddressRegionKHR,
    pub miss_shader_binding_table_buffer: Option<super::buffer::Buffer>,
    pub miss_shader_binding_table: vk::StridedDeviceAddressRegionKHR,
    pub hit_shader_binding_table_buffer: Option<super::buffer::Buffer>,
    pub hit_shader_binding_table: vk::StridedDeviceAddressRegionKHR,
    pub callable_shader_binding_table_buffer: Option<super::buffer::Buffer>,
    pub callable_shader_binding_table: vk::StridedDeviceAddressRegionKHR,
}

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
