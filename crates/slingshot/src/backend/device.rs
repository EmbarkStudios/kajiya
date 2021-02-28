use super::{
    physical_device::{PhysicalDevice, QueueFamily},
    profiler::VkProfilerData,
};
use anyhow::Result;
use ash::{
    extensions::khr,
    version::{DeviceV1_0, InstanceV1_0, InstanceV1_1},
    vk,
};
use gpu_allocator::{AllocatorDebugSettings, VulkanAllocator, VulkanAllocatorCreateDesc};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};

#[allow(dead_code)]
pub struct Queue {
    pub(crate) raw: vk::Queue,
    pub(crate) family: QueueFamily,
}

pub trait DeferredRelease: Copy {
    fn enqueue_release(self, pending: &mut PendingResourceReleases);
}

impl DeferredRelease for vk::DescriptorPool {
    fn enqueue_release(self, pending: &mut PendingResourceReleases) {
        pending.descriptor_pools.push(self);
    }
}

#[derive(Default)]
pub struct PendingResourceReleases {
    pub descriptor_pools: Vec<vk::DescriptorPool>,
}

impl PendingResourceReleases {
    fn release_all(&mut self, device: &ash::Device) {
        unsafe {
            for res in self.descriptor_pools.drain(..) {
                device.destroy_descriptor_pool(res, None);
            }
        }
    }
}

pub struct DeviceFrame {
    //pub(crate) linear_allocator_pool: vk_mem::AllocatorPool,
    pub swapchain_acquired_semaphore: Option<vk::Semaphore>,
    pub rendering_complete_semaphore: Option<vk::Semaphore>,
    pub command_buffer: CommandBuffer,
    pub pending_resource_releases: Mutex<PendingResourceReleases>,
    pub profiler_data: VkProfilerData,
}

pub struct CommandBuffer {
    pub raw: vk::CommandBuffer,
    pub(crate) submit_done_fence: vk::Fence,
    //pool: vk::CommandPool,
}

impl CommandBuffer {
    fn new(device: &ash::Device, queue_family: &QueueFamily) -> Result<Self> {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family.index);

        let pool = unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let cb = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        }[0];

        let submit_done_fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )
        }?;

        Ok(CommandBuffer {
            raw: cb,
            //pool,
            submit_done_fence,
        })
    }
}

impl DeviceFrame {
    pub fn new(
        device: &ash::Device,
        global_allocator: &mut VulkanAllocator,
        queue_family: &QueueFamily,
    ) -> Self {
        Self {
            /*linear_allocator_pool: global_allocator
            .create_pool(&{
                let mut info = vk_mem::AllocatorPoolCreateInfo::default();
                info.flags = vk_mem::AllocatorPoolCreateFlags::LINEAR_ALGORITHM;
                info
            })
            .expect("linear allocator"),*/
            swapchain_acquired_semaphore: None,
            rendering_complete_semaphore: None,
            command_buffer: CommandBuffer::new(device, queue_family).unwrap(),
            pending_resource_releases: Default::default(),
            profiler_data: VkProfilerData::new(device, global_allocator),
        }
    }
}

pub(crate) struct CmdExt {
    pub push_descriptor: khr::PushDescriptor,
}

pub struct Device {
    pub raw: ash::Device,
    pub(crate) pdevice: Arc<PhysicalDevice>,
    pub(crate) instance: Arc<super::instance::Instance>,
    pub(crate) universal_queue: Queue,
    pub(crate) global_allocator: Arc<Mutex<VulkanAllocator>>,
    pub(crate) immutable_samplers: HashMap<SamplerDesc, vk::Sampler>,
    pub(crate) cmd_ext: CmdExt,
    pub(crate) setup_cb: Mutex<CommandBuffer>,
    pub(crate) acceleration_structure_ext: khr::AccelerationStructure,
    pub(crate) ray_tracing_pipeline_ext: khr::RayTracingPipeline,
    // pub(crate) ray_query_ext: khr::RayQuery,
    pub(crate) ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    frame0: Mutex<Arc<DeviceFrame>>,
    frame1: Mutex<Arc<DeviceFrame>>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    fn extension_names(pdevice: &Arc<PhysicalDevice>) -> Vec<*const i8> {
        let mut device_extension_names_raw = vec![
            vk::ExtDescriptorIndexingFn::name().as_ptr(),
            vk::ExtScalarBlockLayoutFn::name().as_ptr(),
            vk::KhrMaintenance1Fn::name().as_ptr(),
            vk::KhrMaintenance2Fn::name().as_ptr(),
            vk::KhrMaintenance3Fn::name().as_ptr(),
            vk::KhrGetMemoryRequirements2Fn::name().as_ptr(),
            vk::ExtDescriptorIndexingFn::name().as_ptr(),
            vk::KhrImagelessFramebufferFn::name().as_ptr(),
            vk::KhrImageFormatListFn::name().as_ptr(),
            //vk::ExtFragmentShaderInterlockFn::name().as_ptr(),
            vk::KhrPushDescriptorFn::name().as_ptr(),
            vk::KhrDescriptorUpdateTemplateFn::name().as_ptr(),
            vk::KhrDrawIndirectCountFn::name().as_ptr(),
        ];

        #[cfg(feature = "ray-tracing")]
        {
            device_extension_names_raw.extend(
                [
                    vk::KhrPipelineLibraryFn::name().as_ptr(),        // rt dep
                    vk::KhrDeferredHostOperationsFn::name().as_ptr(), // rt dep
                    vk::KhrBufferDeviceAddressFn::name().as_ptr(),    // rt dep
                    vk::KhrPipelineLibraryFn::name().as_ptr(),        // rt dep
                    vk::KhrAccelerationStructureFn::name().as_ptr(),
                    vk::KhrRayTracingPipelineFn::name().as_ptr(),
                    vk::KhrRayQueryFn::name().as_ptr(),
                ]
                .iter(),
            );
        }

        if pdevice.presentation_requested {
            device_extension_names_raw.push(khr::Swapchain::name().as_ptr());
        }

        device_extension_names_raw
    }

    pub fn create(pdevice: &Arc<PhysicalDevice>) -> Result<Arc<Self>> {
        let device_extension_names = Self::extension_names(&pdevice);

        let priorities = [1.0];

        let universal_queue = pdevice
            .queue_families
            .iter()
            .filter(|qf| qf.properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .copied()
            .next();

        let universal_queue = if let Some(universal_queue) = universal_queue {
            universal_queue
        } else {
            anyhow::bail!("No suitable render queue found");
        };

        let universal_queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(universal_queue.index)
            .queue_priorities(&priorities)
            .build()];

        let mut scalar_block = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::builder()
            .scalar_block_layout(true)
            .build();

        let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
            .shader_input_attachment_array_dynamic_indexing(true)
            .shader_uniform_texel_buffer_array_dynamic_indexing(true)
            .shader_storage_texel_buffer_array_dynamic_indexing(true)
            .shader_uniform_buffer_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .shader_input_attachment_array_non_uniform_indexing(true)
            .shader_uniform_texel_buffer_array_non_uniform_indexing(true)
            .shader_storage_texel_buffer_array_non_uniform_indexing(true)
            .descriptor_binding_uniform_buffer_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_uniform_texel_buffer_update_after_bind(true)
            .descriptor_binding_storage_texel_buffer_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .build();

        let mut imageless_framebuffer =
            vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::builder()
                .imageless_framebuffer(true)
                .build();

        let mut fragment_shader_interlock =
            vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT::builder()
                .fragment_shader_pixel_interlock(true)
                .build();

        #[cfg(feature = "ray-tracing")]
        let mut acceleration_structure_features =
            ash::vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true)
                //.acceleration_structure_host_commands(true)
                .descriptor_binding_acceleration_structure_update_after_bind(true)
                .build();

        #[cfg(feature = "ray-tracing")]
        let mut ray_tracing_pipeline_features =
            ash::vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder()
                .ray_tracing_pipeline(true)
                .ray_tracing_pipeline_trace_rays_indirect(true)
                .build();

        #[cfg(feature = "ray-tracing")]
        let mut ray_query_features = ash::vk::PhysicalDeviceRayQueryFeaturesKHR::builder()
            .ray_query(true)
            .build();

        let mut get_buffer_device_address_features =
            ash::vk::PhysicalDeviceBufferDeviceAddressFeatures {
                buffer_device_address: 1,
                ..Default::default()
            };

        unsafe {
            let instance = &pdevice.instance.raw;

            let mut features2 = vk::PhysicalDeviceFeatures2::default();
            instance
                .fp_v1_1()
                .get_physical_device_features2(pdevice.raw, &mut features2);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&universal_queue_info)
                .enabled_extension_names(&device_extension_names)
                .push_next(&mut features2)
                .push_next(&mut scalar_block)
                .push_next(&mut descriptor_indexing)
                .push_next(&mut imageless_framebuffer)
                .push_next(&mut fragment_shader_interlock)
                .push_next(&mut get_buffer_device_address_features);

            #[cfg(feature = "ray-tracing")]
            let device_create_info = device_create_info
                .push_next(&mut acceleration_structure_features)
                .push_next(&mut ray_tracing_pipeline_features)
                .push_next(&mut ray_query_features)
                .build();

            let device = instance
                .create_device(pdevice.raw, &device_create_info, None)
                .unwrap();

            info!("Created a Vulkan device");

            let mut global_allocator = VulkanAllocator::new(&VulkanAllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice.raw,
                debug_settings: AllocatorDebugSettings {
                    log_leaks_on_shutdown: false,
                    log_memory_information: true,
                    log_allocations: true,
                    ..Default::default()
                },
            });

            let universal_queue = Queue {
                raw: device.get_device_queue(universal_queue.index, 0),
                family: universal_queue,
            };

            let frame0 = DeviceFrame::new(&device, &mut global_allocator, &universal_queue.family);
            let frame1 = DeviceFrame::new(&device, &mut global_allocator, &universal_queue.family);

            let immutable_samplers = Self::create_samplers(&device);
            let cmd_ext = CmdExt {
                push_descriptor: khr::PushDescriptor::new(&pdevice.instance.raw, &device),
            };

            let setup_cb = CommandBuffer::new(&device, &universal_queue.family).unwrap();

            let acceleration_structure_ext =
                khr::AccelerationStructure::new(&pdevice.instance.raw, &device);
            let ray_tracing_pipeline_ext =
                khr::RayTracingPipeline::new(&pdevice.instance.raw, &device);
            //let ray_query_ext = khr::RayQuery::new(&pdevice.instance.raw, &device);
            let ray_tracing_pipeline_properties =
                khr::RayTracingPipeline::get_properties(&pdevice.instance.raw, pdevice.raw);

            Ok(Arc::new(Device {
                pdevice: pdevice.clone(),
                instance: pdevice.instance.clone(),
                raw: device,
                universal_queue,
                global_allocator: Arc::new(Mutex::new(global_allocator)),
                immutable_samplers,
                cmd_ext,
                setup_cb: Mutex::new(setup_cb),
                acceleration_structure_ext,
                ray_tracing_pipeline_ext,
                // ray_query_ext,
                ray_tracing_pipeline_properties,
                frame0: Mutex::new(Arc::new(frame0)),
                frame1: Mutex::new(Arc::new(frame1)),
            }))
        }
    }

    fn create_samplers(device: &ash::Device) -> HashMap<SamplerDesc, vk::Sampler> {
        let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
        let mipmap_modes = [
            vk::SamplerMipmapMode::NEAREST,
            vk::SamplerMipmapMode::LINEAR,
        ];
        let address_modes = [
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ];

        let mut result = HashMap::new();

        for &texel_filter in &texel_filters {
            for &mipmap_mode in &mipmap_modes {
                for &address_modes in &address_modes {
                    let anisotropy_enable = texel_filter == vk::Filter::LINEAR;

                    result.insert(
                        SamplerDesc {
                            texel_filter,
                            mipmap_mode,
                            address_modes,
                        },
                        unsafe {
                            device.create_sampler(
                                &vk::SamplerCreateInfo::builder()
                                    .mag_filter(texel_filter)
                                    .min_filter(texel_filter)
                                    .mipmap_mode(mipmap_mode)
                                    .address_mode_u(address_modes)
                                    .address_mode_v(address_modes)
                                    .address_mode_w(address_modes)
                                    .max_lod(vk::LOD_CLAMP_NONE)
                                    .max_anisotropy(16.0)
                                    .anisotropy_enable(anisotropy_enable)
                                    .build(),
                                None,
                            )
                        }
                        .expect("create_sampler"),
                    );
                }
            }
        }

        result
    }

    pub fn get_sampler(&self, desc: SamplerDesc) -> vk::Sampler {
        *self
            .immutable_samplers
            .get(&desc)
            .unwrap_or_else(|| panic!("Sampler not found: {:?}", desc))
    }

    pub fn current_frame(&self) -> Arc<DeviceFrame> {
        self.frame0.lock().clone()
    }

    pub fn defer_release(&self, resource: impl DeferredRelease) {
        resource.enqueue_release(&mut *self.current_frame().pending_resource_releases.lock());
    }

    pub fn with_setup_cb(&self, callback: impl FnOnce(vk::CommandBuffer)) {
        let cb = self.setup_cb.lock();

        unsafe {
            self.raw
                .begin_command_buffer(
                    cb.raw,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }

        callback(cb.raw);

        unsafe {
            self.raw.end_command_buffer(cb.raw).unwrap();

            let submit_info =
                vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&cb.raw));

            self.raw
                .queue_submit(
                    self.universal_queue.raw,
                    &[submit_info.build()],
                    vk::Fence::null(),
                )
                .expect("queue submit failed.");

            self.raw.device_wait_idle().unwrap();
        }
    }

    pub fn finish_frame(&self, frame: Arc<DeviceFrame>) {
        drop(frame);

        let mut frame0 = self.frame0.lock();
        let mut frame1 = self.frame1.lock();

        let frame0: &mut DeviceFrame = Arc::get_mut(&mut frame0).unwrap_or_else(|| {
            panic!("Unable to finish frame: frame data is being held by user code")
        });
        let frame1: &mut DeviceFrame = Arc::get_mut(&mut frame1).unwrap();

        std::mem::swap(frame0, frame1);

        // Wait for the the GPU to be done with the previously submitted frame,
        // so that we can access its data again
        unsafe {
            self.raw
                .wait_for_fences(
                    std::slice::from_ref(&frame0.command_buffer.submit_done_fence),
                    true,
                    std::u64::MAX,
                )
                .expect("Wait for fence failed.");

            self.raw
                .reset_command_buffer(
                    frame0.command_buffer.raw,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .unwrap();
        }

        // Report GPU timings
        {
            let (query_ids, timing_pairs) = frame0.profiler_data.retrieve_previous_result();

            let ns_per_tick = self.pdevice.properties.limits.timestamp_period;

            crate::gpu_profiler::report_durations_ticks(
                ns_per_tick,
                timing_pairs.chunks_exact(2).enumerate().map(
                    |(pair_idx, chunk)| -> (crate::gpu_profiler::GpuProfilerQueryId, u64) {
                        (query_ids[pair_idx], chunk[1] - chunk[0])
                    },
                ),
            );
        }

        frame0
            .pending_resource_releases
            .get_mut()
            .release_all(&self.raw);
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        self.pdevice.as_ref()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
        }
    }
}

// TODO
/*impl Drop for Device {
    fn drop(&mut self) {
        let mut frame0 = self.frame0.lock();
        let mut frame1 = self.frame1.lock();

        let frame0: &mut DeviceFrame = Arc::get_mut(&mut frame0).unwrap_or_else(|| {
            panic!("Unable to deallocate DeviceFrame: frame data is being held by user code")
        });
        let frame1: &mut DeviceFrame = Arc::get_mut(&mut frame1).unwrap_or_else(|| {
            panic!("Unable to deallocate DeviceFrame: frame data is being held by user code(2)")
        });

        /*self.global_allocator
            .destroy_pool(&frame0.linear_allocator_pool)
            .unwrap();

        self.global_allocator
            .destroy_pool(&frame1.linear_allocator_pool)
            .unwrap();*/
    }
}*/

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_modes: vk::SamplerAddressMode,
}
