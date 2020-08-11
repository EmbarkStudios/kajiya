mod device;
mod instance;
mod logging;
mod physical_device;
mod surface;
mod swapchain;

use ash::{version::DeviceV1_0, vk};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;
use swapchain::{Swapchain, SwapchainDesc};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct WindowConfig {
    width: u32,
    height: u32,
}

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

fn main() -> anyhow::Result<()> {
    logging::set_up_logging()?;

    let event_loop = EventLoop::new();

    let window_cfg = WindowConfig {
        width: 1280,
        height: 720,
    };

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("vicki")
            .with_inner_size(winit::dpi::LogicalSize::new(
                window_cfg.width as f64,
                window_cfg.height as f64,
            ))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = instance::Instance::builder()
        .required_extensions(ash_window::enumerate_required_extensions(&*window).unwrap())
        .build()?;
    let surface = surface::Surface::create(&instance, &*window)?;

    use physical_device::*;
    let physical_devices =
        enumerate_physical_devices(&instance)?.with_presentation_support(&surface);

    info!("Available physical devices: {:#?}", physical_devices);

    let physical_device = Arc::new(
        physical_devices
            .into_iter()
            .next()
            .expect("valid physical device"),
    );

    let device = device::Device::create(&physical_device)?;
    let surface_formats = Swapchain::enumerate_surface_formats(&device, &surface)?;

    info!("Available surface formats: {:#?}", surface_formats);

    let mut swapchain = Swapchain::new(
        &device,
        &surface,
        SwapchainDesc {
            format: select_surface_format(surface_formats).expect("suitable surface format"),
            dims: vk::Extent2D {
                width: window_cfg.width,
                height: window_cfg.height,
            },
            vsync: true,
        },
    )?;

    event_loop.run(move |event, _, control_flow| {
        // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
        // dispatched any events. This is ideal for games and similar applications.
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                // Application update code.

                let swapchain_image = swapchain
                    .acquire_next_image()
                    .ok()
                    .expect("swapchain image");

                let submit_info = vk::SubmitInfo::builder()
                    .wait_semaphores(std::slice::from_ref(&swapchain_image.acquire_semaphore))
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]);

                /*unsafe {
                    device
                        .raw
                        .queue_submit(
                            device.universal_queue.raw,
                            &[submit_info.build()],
                            vk::Fence::default(),
                        )
                        .expect("queue submit failed.");
                }*/

                swapchain.present_image(swapchain_image, &[]);

                info!("Swapchain image acquired");
            }
            _ => (),
        }
    })
}
