use crate::physical_device::{PhysicalDevice, QueueFamily};
use anyhow::Result;
use ash::{
    extensions::khr,
    version::{DeviceV1_0, InstanceV1_0, InstanceV1_1},
    vk,
};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;

#[allow(dead_code)]
pub struct Queue {
    pub(crate) raw: vk::Queue,
    pub(crate) family: QueueFamily,
}

pub struct DeviceFrame {
    // TODO: linear allocator
    pub swapchain_acquired_semaphore: Option<vk::Semaphore>,
    pub rendering_complete_semaphore: Option<vk::Semaphore>,
}

pub struct Device {
    pub(crate) pdevice: Arc<PhysicalDevice>,
    pub(crate) instance: Arc<crate::instance::Instance>,
    pub(crate) raw: ash::Device,
    pub(crate) universal_queue: Queue,
    pub(crate) allocator: vk_mem::Allocator,
}

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
            vk::ExtFragmentShaderInterlockFn::name().as_ptr(),
            khr::RayTracing::name().as_ptr(),
            // rt dep
            ash::vk::KhrPipelineLibraryFn::name().as_ptr(),
            // rt dep
            ash::vk::KhrDeferredHostOperationsFn::name().as_ptr(),
        ];

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
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_update_unused_while_pending(true)
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true)
            .shader_uniform_texel_buffer_array_dynamic_indexing(true)
            .shader_uniform_texel_buffer_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .build();

        let mut imageless_framebuffer =
            vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::builder()
                .imageless_framebuffer(true)
                .build();

        let mut fragment_shader_interlock =
            vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT::builder()
                .fragment_shader_pixel_interlock(true)
                .build();

        let mut ray_tracing_features = ash::vk::PhysicalDeviceRayTracingFeaturesKHR::default();
        ray_tracing_features.ray_tracing = 1;
        ray_tracing_features.ray_query = 1;

        let mut get_buffer_device_address_features =
            ash::vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR::default();
        get_buffer_device_address_features.buffer_device_address = 1;

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
                .push_next(&mut ray_tracing_features)
                .push_next(&mut get_buffer_device_address_features)
                .build();

            let device = instance
                .create_device(pdevice.raw, &device_create_info, None)
                .unwrap();

            info!("Created a Vulkan device");

            let allocator_info = vk_mem::AllocatorCreateInfo {
                physical_device: pdevice.raw,
                device: device.clone(),
                instance: instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::NONE,
                preferred_large_heap_block_size: 0,
                frame_in_use_count: 0,
                heap_size_limits: None,
            };

            let allocator = vk_mem::Allocator::new(&allocator_info)
                .expect("Failed to initialize the Vulkan Memory Allocator");

            let universal_queue = Queue {
                raw: device.get_device_queue(universal_queue.index, 0),
                family: universal_queue,
            };

            Ok(Arc::new(Device {
                pdevice: pdevice.clone(),
                instance: pdevice.instance.clone(),
                raw: device,
                universal_queue,
                allocator,
            }))
        }
    }
}
