use crate::{
    instance::Instance,
    physical_device::{PhysicalDevice, QueueFamily},
    surface::Surface,
};
use anyhow::Result;
use ash::{
    extensions::{ext, khr},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1},
    vk,
};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

#[derive(Clone, Copy, Default)]
pub struct SwapchainDesc {
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
    pub vsync: bool,
}

pub struct Swapchain {
    pub(crate) fns: khr::Swapchain,
    pub(crate) raw: vk::SwapchainKHR,
}
