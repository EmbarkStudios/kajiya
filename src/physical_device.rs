use crate::{instance::Instance, surface::Surface};
use anyhow::Result;
use ash::{version::InstanceV1_0, vk};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{ffi::CStr, sync::Arc};

/// Properties of the physical device.
#[derive(Clone, Debug)]
pub struct PhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub device_name: String,
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: vk::PhysicalDeviceLimits,
    pub sparse_properties: vk::PhysicalDeviceSparseProperties,
}

#[derive(Copy, Clone)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

pub struct PhysicalDevice {
    pub(crate) instance: Arc<Instance>,
    pub(crate) raw: vk::PhysicalDevice,
    pub(crate) queue_families: Vec<QueueFamily>,
    pub(crate) properties: PhysicalDeviceProperties,
    pub(crate) presentation_requested: bool,
}

impl std::fmt::Debug for PhysicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PhysicalDevice {{ {:#?} }}", self.properties)
    }
}

pub fn enumerate_physical_devices(instance: &Arc<Instance>) -> Result<Vec<PhysicalDevice>> {
    unsafe {
        let pdevices = instance.raw.enumerate_physical_devices()?;

        Ok(pdevices
            .into_iter()
            .map(|pdevice| {
                let properties = instance.raw.get_physical_device_properties(pdevice);
                let properties = PhysicalDeviceProperties {
                    api_version: properties.api_version,
                    driver_version: properties.driver_version,
                    vendor_id: properties.vendor_id,
                    device_id: properties.device_id,
                    device_type: properties.device_type,
                    device_name: CStr::from_ptr(&properties.device_name[0])
                        .to_str()
                        .unwrap()
                        .to_string(),
                    pipeline_cache_uuid: properties.pipeline_cache_uuid,
                    limits: properties.limits,
                    sparse_properties: properties.sparse_properties,
                };

                let queue_families = instance
                    .raw
                    .get_physical_device_queue_family_properties(pdevice)
                    .into_iter()
                    .enumerate()
                    .map(|(index, properties)| QueueFamily {
                        index: index as _,
                        properties,
                    })
                    .collect();

                PhysicalDevice {
                    raw: pdevice,
                    queue_families,
                    properties,
                    presentation_requested: true,
                    instance: instance.clone(),
                }
            })
            .collect())
    }
}

pub trait PhysicalDeviceList {
    fn with_presentation_support(self, surface: &Surface) -> Self;
}

impl PhysicalDeviceList for Vec<PhysicalDevice> {
    fn with_presentation_support(self, surface: &Surface) -> Self {
        self.into_iter()
            .filter_map(|mut pdevice| {
                pdevice.presentation_requested = true;

                let supports_presentation =
                    pdevice
                        .queue_families
                        .iter()
                        .enumerate()
                        .any(|(queue_index, info)| unsafe {
                            info.properties
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS)
                                && surface
                                    .fns
                                    .get_physical_device_surface_support(
                                        pdevice.raw,
                                        queue_index as u32,
                                        surface.raw,
                                    )
                                    .unwrap()
                        });

                if supports_presentation {
                    Some(pdevice)
                } else {
                    None
                }
            })
            .collect()
    }
}
