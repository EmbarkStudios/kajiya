pub mod barrier;
pub mod buffer;
pub mod device;
pub mod error;
pub mod image;
pub mod instance;
pub mod physical_device;
mod profiler;
pub mod ray_tracing;
pub mod shader;
pub mod surface;
pub mod swapchain;

use ash::vk;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use raw_window_handle::HasRawWindowHandle;
use std::sync::Arc;

fn select_surface_format(formats: Vec<vk::SurfaceFormatKHR>) -> Option<vk::SurfaceFormatKHR> {
    let preferred = vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_UNORM,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };

    if formats.contains(&preferred) {
        Some(preferred)
    } else {
        None
    }
}

pub struct RenderBackend {
    pub device: Arc<device::Device>,
    pub surface: Arc<surface::Surface>,
    pub swapchain: swapchain::Swapchain,
}

#[derive(Clone, Copy)]
pub struct RenderBackendConfig {
    pub swapchain_extent: [u32; 2],
    pub vsync: bool,
    pub graphics_debugging: bool,
    pub device_index: Option<usize>,
}

impl RenderBackend {
    pub fn new(
        window: &impl HasRawWindowHandle,
        config: RenderBackendConfig,
    ) -> anyhow::Result<Self> {
        let instance = instance::Instance::builder()
            .required_extensions(ash_window::enumerate_required_extensions(window).unwrap())
            .graphics_debugging(config.graphics_debugging)
            .build()?;
        let surface = surface::Surface::create(&instance, window)?;

        use physical_device::*;
        let physical_devices =
            enumerate_physical_devices(&instance)?.with_presentation_support(&surface);

        info!(
            "Available physical devices: {:#?}",
            physical_devices
                .iter()
                .map(|dev| unsafe {
                    ::std::ffi::CStr::from_ptr(
                        dev.properties.device_name.as_ptr() as *const std::os::raw::c_char
                    )
                })
                .collect::<Vec<_>>()
        );

        let physical_device = Arc::new(if let Some(device_index) = config.device_index {
            physical_devices.into_iter().nth(device_index).unwrap()
        } else {
            physical_devices
                .into_iter()
                // If there are multiple devices with the same score, `max_by_key` would choose the last,
                // and we want to preserve the order of devices from `enumerate_physical_devices`.
                .rev()
                .max_by_key(|device| match device.properties.device_type {
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 200,
                    vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0,
                })
                .unwrap()
        });

        info!("Selected physical device: {:#?}", *physical_device);

        let device = device::Device::create(&physical_device)?;
        let surface_formats = swapchain::Swapchain::enumerate_surface_formats(&device, &surface)?;

        info!("Available surface formats: {:#?}", surface_formats);

        let swapchain = swapchain::Swapchain::new(
            &device,
            &surface,
            swapchain::SwapchainDesc {
                format: select_surface_format(surface_formats).expect("suitable surface format"),
                dims: vk::Extent2D {
                    width: config.swapchain_extent[0],
                    height: config.swapchain_extent[1],
                },
                vsync: config.vsync,
            },
        )?;

        Ok(Self {
            device,
            surface,
            swapchain,
        })
    }

    /*fn maintain(&mut self) {
        self.images.maintain();
    }*/
}
