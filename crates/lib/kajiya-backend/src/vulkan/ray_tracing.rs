use std::sync::Arc;

use crate::{dynamic_constants::DynamicConstants, MAX_DESCRIPTOR_SETS};

use super::{
    device::Device,
    shader::{
        merge_shader_stage_layouts, DescriptorSetLayoutOpts, PipelineShader, ShaderPipelineCommon,
        ShaderPipelineStage,
    },
};
use anyhow::{Context, Result};
use ash::vk;
use byte_slice_cast::AsSliceOf;
use bytes::Bytes;
use glam::Affine3A;
use parking_lot::Mutex;

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

#[derive(Clone)]
pub struct RayTracingInstanceDesc {
    pub blas: Arc<RayTracingAcceleration>,
    pub transformation: Affine3A,
    pub mesh_index: u32,
}

#[derive(Clone)]
pub struct RayTracingTopAccelerationDesc {
    pub instances: Vec<RayTracingInstanceDesc>,
    pub preallocate_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct RayTracingBottomAccelerationDesc {
    pub geometries: Vec<RayTracingGeometryDesc>,
}

#[derive(Clone, Debug)]
pub struct RayTracingShaderTableDesc {
    pub raygen_entry_count: u32,
    pub hit_entry_count: u32,
    pub miss_entry_count: u32,
}

pub struct RayTracingAcceleration {
    pub raw: vk::AccelerationStructureKHR,
    backing_buffer: super::buffer::Buffer,
}

#[derive(Clone)]
pub struct RayTracingAccelerationScratchBuffer {
    buffer: Arc<Mutex<super::buffer::Buffer>>,
}

impl Device {
    pub fn create_ray_tracing_acceleration_scratch_buffer(
        &self,
    ) -> Result<RayTracingAccelerationScratchBuffer> {
        const INITIAL_SIZE: usize = 1024 * 1024 * 144;

        let buffer = self
            .create_buffer(
                super::buffer::BufferDesc {
                    size: INITIAL_SIZE,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    mapped: false,
                },
                None,
            )
            .context("Acceleration structure scratch buffer")?;

        Ok(RayTracingAccelerationScratchBuffer {
            buffer: Arc::new(Mutex::new(buffer)),
        })
    }

    pub fn create_ray_tracing_bottom_acceleration(
        &self,
        desc: &RayTracingBottomAccelerationDesc,
        scratch_buffer: &RayTracingAccelerationScratchBuffer,
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

        let preallocate_bytes = 0;
        self.create_ray_tracing_acceleration(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            geometry_info,
            &build_range_infos,
            &max_primitive_counts,
            preallocate_bytes,
            scratch_buffer,
        )
    }

    pub fn create_ray_tracing_top_acceleration(
        &self,
        desc: &RayTracingTopAccelerationDesc,
        scratch_buffer: &RayTracingAccelerationScratchBuffer,
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
                            &ash::vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                                .acceleration_structure(desc.blas.raw)
                                .build(),
                        )
                };

                let transform = [
                    desc.transformation.x_axis.x,
                    desc.transformation.y_axis.x,
                    desc.transformation.z_axis.x,
                    desc.transformation.translation.x,
                    desc.transformation.x_axis.y,
                    desc.transformation.y_axis.y,
                    desc.transformation.z_axis.y,
                    desc.transformation.translation.y,
                    desc.transformation.x_axis.z,
                    desc.transformation.y_axis.z,
                    desc.transformation.z_axis.z,
                    desc.transformation.translation.z,
                ];

                GeometryInstance::new(
                    transform,
                    desc.mesh_index, /* instance id */
                    0xff,
                    0,
                    /*ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                    | */
                    ash::vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE,
                    blas_address,
                )
            })
            .collect();

        let instance_buffer_size = std::mem::size_of::<GeometryInstance>() * instances.len().max(1);
        let instance_buffer = self
            .create_buffer(
                super::buffer::BufferDesc {
                    size: instance_buffer_size,
                    usage: ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    mapped: false,
                },
                unsafe {
                    (!instances.is_empty()).then(|| {
                        std::slice::from_raw_parts(
                            instances.as_ptr() as *const u8,
                            instance_buffer_size,
                        )
                    })
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
            desc.preallocate_bytes,
            scratch_buffer,
        )
    }

    fn create_ray_tracing_acceleration(
        &self,
        ty: vk::AccelerationStructureTypeKHR,
        mut geometry_info: vk::AccelerationStructureBuildGeometryInfoKHR,
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
        max_primitive_counts: &[u32],
        preallocate_bytes: usize,
        scratch_buffer: &RayTracingAccelerationScratchBuffer,
    ) -> Result<RayTracingAcceleration> {
        let memory_requirements = unsafe {
            self.acceleration_structure_ext
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &geometry_info,
                    max_primitive_counts,
                )
        };

        log::info!(
            "Acceleration structure size: {}, scratch size: {}",
            memory_requirements.acceleration_structure_size,
            memory_requirements.build_scratch_size
        );

        let backing_buffer_size: usize =
            preallocate_bytes.max(memory_requirements.acceleration_structure_size as usize);

        let accel_buffer = self
            .create_buffer(
                super::buffer::BufferDesc {
                    size: backing_buffer_size,
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    mapped: false,
                },
                None,
            )
            .context("Acceleration structure buffer")?;

        let accel_info = ash::vk::AccelerationStructureCreateInfoKHR::builder()
            .ty(ty)
            .buffer(accel_buffer.raw)
            .size(backing_buffer_size as u64)
            .build();

        unsafe {
            let accel_raw = self
                .acceleration_structure_ext
                .create_acceleration_structure(&accel_info, None)
                .context("create_acceleration_structure")?;

            let scratch_buffer = scratch_buffer.buffer.lock();
            assert!(
                memory_requirements.build_scratch_size as usize <= scratch_buffer.desc.size,
                "todo: resize scratch"
            );

            /*let scratch_buffer = self
            .create_buffer(
                super::buffer::BufferDesc {
                    size: memory_requirements.build_scratch_size as _,
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    mapped: false,
                },
                None,
            )
            .context("Acceleration structure scratch buffer")?;*/

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
                backing_buffer: accel_buffer,
            })
        }
    }

    pub fn fill_ray_tracing_instance_buffer(
        &self,
        dynamic_constants: &mut DynamicConstants,
        instances: &[RayTracingInstanceDesc],
    ) -> vk::DeviceAddress {
        let instance_buffer_address = dynamic_constants.current_device_address(self);

        dynamic_constants.push_from_iter(instances.iter().map(|desc| {
            let blas_address = unsafe {
                self.acceleration_structure_ext
                    .get_acceleration_structure_device_address(
                        &ash::vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                            .acceleration_structure(desc.blas.raw)
                            .build(),
                    )
            };

            let transform = [
                desc.transformation.x_axis.x,
                desc.transformation.y_axis.x,
                desc.transformation.z_axis.x,
                desc.transformation.translation.x,
                desc.transformation.x_axis.y,
                desc.transformation.y_axis.y,
                desc.transformation.z_axis.y,
                desc.transformation.translation.y,
                desc.transformation.x_axis.z,
                desc.transformation.y_axis.z,
                desc.transformation.z_axis.z,
                desc.transformation.translation.z,
            ];

            GeometryInstance::new(
                transform,
                desc.mesh_index, /* instance id */
                0xff,
                0,
                /*ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                | */
                ash::vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE,
                blas_address,
            )
        }));

        instance_buffer_address
    }

    pub fn rebuild_ray_tracing_top_acceleration(
        &self,
        cb: vk::CommandBuffer,
        instance_buffer_address: vk::DeviceAddress,
        instance_count: usize,
        tlas: &RayTracingAcceleration,
        scratch_buffer: &RayTracingAccelerationScratchBuffer,
    ) {
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
            .primitive_count(instance_count as _)
            .build()];

        let geometry_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .build();

        let max_primitive_counts = [instance_count as u32];

        // Create top-level acceleration structure

        self.rebuild_ray_tracing_acceleration(
            cb,
            geometry_info,
            &build_range_infos,
            &max_primitive_counts,
            tlas,
            scratch_buffer,
        )
    }

    fn rebuild_ray_tracing_acceleration(
        &self,
        cb: vk::CommandBuffer,
        mut geometry_info: vk::AccelerationStructureBuildGeometryInfoKHR,
        build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR],
        max_primitive_counts: &[u32],
        accel: &RayTracingAcceleration,
        scratch_buffer: &RayTracingAccelerationScratchBuffer,
    ) {
        let memory_requirements = unsafe {
            self.acceleration_structure_ext
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &geometry_info,
                    max_primitive_counts,
                )
        };

        assert!(
            memory_requirements.acceleration_structure_size as usize
                <= accel.backing_buffer.desc.size,
            "todo: backing"
        );

        let scratch_buffer = scratch_buffer.buffer.lock();

        assert!(
            memory_requirements.build_scratch_size as usize <= scratch_buffer.desc.size,
            "todo: scratch"
        );

        unsafe {
            geometry_info.dst_acceleration_structure = accel.raw;
            geometry_info.scratch_data = ash::vk::DeviceOrHostAddressKHR {
                device_address: self.raw.get_buffer_device_address(
                    &ash::vk::BufferDeviceAddressInfo::builder().buffer(scratch_buffer.raw),
                ),
            };

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
                .context("get_ray_tracing_shader_group_handles")?
        };

        let prog_size = shader_group_handle_size;

        let create_binding_table = |entry_offset: u32,
                                    entry_count: u32|
         -> Result<Option<crate::vulkan::buffer::Buffer>> {
            if 0 == entry_count {
                return Ok(None);
            }

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

            Ok(Some(
                self.create_buffer(
                    super::buffer::BufferDesc {
                        size: shader_binding_table_data.len(),
                        usage: vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
                        mapped: false,
                    },
                    Some(&shader_binding_table_data),
                )
                .context("SBT sub-buffer")?,
            ))
        };

        let raygen_shader_binding_table = create_binding_table(0, desc.raygen_entry_count)?;
        let miss_shader_binding_table =
            create_binding_table(desc.raygen_entry_count, desc.miss_entry_count)?;
        let hit_shader_binding_table = create_binding_table(
            desc.raygen_entry_count + desc.miss_entry_count,
            desc.hit_entry_count,
        )?;

        Ok(RayTracingShaderTable {
            raygen_shader_binding_table: vk::StridedDeviceAddressRegionKHR {
                device_address: raygen_shader_binding_table
                    .as_ref()
                    .map(|b| b.device_address(self))
                    .unwrap_or(0),
                stride: prog_size as u64,
                size: (prog_size * desc.raygen_entry_count as usize) as u64,
            },
            raygen_shader_binding_table_buffer: raygen_shader_binding_table,
            miss_shader_binding_table: vk::StridedDeviceAddressRegionKHR {
                device_address: miss_shader_binding_table
                    .as_ref()
                    .map(|b| b.device_address(self))
                    .unwrap_or(0),
                stride: prog_size as u64,
                size: (prog_size * desc.miss_entry_count as usize) as u64,
            },
            miss_shader_binding_table_buffer: miss_shader_binding_table,
            hit_shader_binding_table: vk::StridedDeviceAddressRegionKHR {
                device_address: hit_shader_binding_table
                    .as_ref()
                    .map(|b| b.device_address(self))
                    .unwrap_or(0),
                stride: prog_size as u64,
                size: (prog_size * desc.hit_entry_count as usize) as u64,
            },
            hit_shader_binding_table_buffer: hit_shader_binding_table,
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

#[derive(Clone)]
pub struct RayTracingPipelineDesc {
    pub descriptor_set_opts: [Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS],
    pub max_pipeline_ray_recursion_depth: u32,
}

impl Default for RayTracingPipelineDesc {
    fn default() -> Self {
        Self {
            max_pipeline_ray_recursion_depth: 1,
            descriptor_set_opts: Default::default(),
        }
    }
}

impl RayTracingPipelineDesc {
    pub fn max_pipeline_ray_recursion_depth(
        mut self,
        max_pipeline_ray_recursion_depth: u32,
    ) -> Self {
        self.max_pipeline_ray_recursion_depth = max_pipeline_ray_recursion_depth;
        self
    }
}

pub fn create_ray_tracing_pipeline(
    device: &Device,
    shaders: &[PipelineShader<Bytes>],
    desc: &RayTracingPipelineDesc,
) -> anyhow::Result<RayTracingPipeline> {
    let stage_layouts = shaders
        .iter()
        .map(|desc| {
            rspirv_reflect::Reflection::new_from_spirv(&desc.code)
                .unwrap_or_else(|err| panic!("Failed compiling shader {:?}:\n{:?}", desc.desc, err))
                .get_descriptor_sets()
                .unwrap()
        })
        .collect::<Vec<_>>();

    //log::info!("{:#?}", stage_layouts);

    let (descriptor_set_layouts, set_layout_info) = super::shader::create_descriptor_set_layouts(
        device,
        &merge_shader_stage_layouts(stage_layouts),
        vk::ShaderStageFlags::ALL,
        //desc.descriptor_set_layout_flags.unwrap_or(&[]),  // TODO: merge flags
        &desc.descriptor_set_opts,
    );

    unsafe {
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .build();

        let pipeline_layout = device
            .raw
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let mut shader_groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> = Vec::new();
        let mut shader_stages: Vec<vk::PipelineShaderStageCreateInfo> = Vec::new();

        // Keep entry point names alive, since build() forgets references.
        let mut entry_points: Vec<std::ffi::CString> = Vec::new();

        let mut raygen_entry_count = 0;
        let mut miss_entry_count = 0;
        let mut hit_entry_count = 0;

        let create_shader_module =
            |desc: &PipelineShader<Bytes>| -> (ash::vk::ShaderModule, String) {
                let shader_info = vk::ShaderModuleCreateInfo::builder()
                    .code(desc.code.as_slice_of::<u32>().unwrap());

                let shader_module = device
                    .raw
                    .create_shader_module(&shader_info, None)
                    .expect("Shader module error");

                (shader_module, desc.desc.entry.clone())
            };

        let mut prev_stage: Option<ShaderPipelineStage> = None;

        for desc in shaders {
            let group_idx = shader_stages.len();

            match desc.desc.stage {
                ShaderPipelineStage::RayGen => {
                    assert!(prev_stage == None || prev_stage == Some(ShaderPipelineStage::RayGen));
                    raygen_entry_count += 1;

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
                        .general_shader(group_idx as _)
                        .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .intersection_shader(ash::vk::SHADER_UNUSED_KHR)
                        .build();

                    shader_stages.push(stage);
                    shader_groups.push(group);
                }
                ShaderPipelineStage::RayMiss => {
                    assert!(
                        prev_stage == Some(ShaderPipelineStage::RayGen)
                            || prev_stage == Some(ShaderPipelineStage::RayMiss)
                    );
                    miss_entry_count += 1;

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
                        .general_shader(group_idx as _)
                        .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .intersection_shader(ash::vk::SHADER_UNUSED_KHR)
                        .build();

                    shader_stages.push(stage);
                    shader_groups.push(group);
                }
                ShaderPipelineStage::RayClosestHit => {
                    assert!(
                        prev_stage == Some(ShaderPipelineStage::RayMiss)
                            || prev_stage == Some(ShaderPipelineStage::RayClosestHit)
                    );
                    hit_entry_count += 1;

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
                        .closest_hit_shader(group_idx as _)
                        .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                        .intersection_shader(ash::vk::SHADER_UNUSED_KHR)
                        .build();

                    shader_stages.push(stage);
                    shader_groups.push(group);
                }
                _ => unimplemented!(),
            }

            prev_stage = Some(desc.desc.stage);
        }

        assert!(raygen_entry_count > 0);
        assert!(miss_entry_count > 0);

        let pipeline = device
            .ray_tracing_pipeline_ext
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[ash::vk::RayTracingPipelineCreateInfoKHR::builder()
                    .stages(&shader_stages)
                    .groups(&shader_groups)
                    .max_pipeline_ray_recursion_depth(desc.max_pipeline_ray_recursion_depth) // TODO
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
                    raygen_entry_count,
                    hit_entry_count,
                    miss_entry_count,
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
