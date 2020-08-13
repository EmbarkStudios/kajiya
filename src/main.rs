mod backend;
mod bytes;
mod logging;
mod mesh;

use backend::{
    barrier::*,
    swapchain::{Swapchain, SwapchainDesc},
};

use ash::{version::DeviceV1_0, vk};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use mesh::LoadGltfScene;
use std::sync::Arc;
use turbosloth::*;
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

struct Renderer {
    //instance: Arc<backend::instance::Instance>,
    //surface: Arc<backend::surface::Surface>,
    device: Arc<backend::device::Device>,
    swapchain: backend::swapchain::Swapchain,
}

impl Renderer {
    fn new(window: &winit::window::Window, window_cfg: &WindowConfig) -> anyhow::Result<Self> {
        let instance = backend::instance::Instance::builder()
            .required_extensions(ash_window::enumerate_required_extensions(&*window).unwrap())
            .graphics_debugging(true)
            .build()?;
        let surface = backend::surface::Surface::create(&instance, &*window)?;

        use backend::physical_device::*;
        let physical_devices =
            enumerate_physical_devices(&instance)?.with_presentation_support(&surface);

        info!("Available physical devices: {:#?}", physical_devices);

        let physical_device = Arc::new(
            physical_devices
                .into_iter()
                .next()
                .expect("valid physical device"),
        );

        let device = backend::device::Device::create(&physical_device)?;
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

        Ok(Self {
            //instance,
            //surface,
            device,
            swapchain,
        })
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

    let mut renderer = Renderer::new(&*window, &window_cfg)?;

    let (present_pipeline_layout, present_pipeline) =
        backend::presentation::create_present_descriptor_set_and_pipeline(&*renderer.device);

    let lazy_cache = LazyCache::create();

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
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                // Note: this can be done at the end of the frame, not at the start.
                // The image can be acquired just in time for a blit into it,
                // after all the other rendering commands have been recorded.
                let swapchain_image = renderer
                    .swapchain
                    .acquire_next_image()
                    .ok()
                    .expect("swapchain image");

                let current_frame = renderer.device.current_frame();
                let cb = &current_frame.command_buffer;

                unsafe {
                    renderer
                        .device
                        .raw
                        .begin_command_buffer(
                            cb.raw,
                            &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )
                        .unwrap();

                    record_image_barrier(
                        &renderer.device.raw,
                        cb.raw,
                        ImageBarrier::new(
                            swapchain_image.image,
                            vk_sync::AccessType::Present,
                            vk_sync::AccessType::ComputeShaderWrite,
                        )
                        .with_discard(true),
                    );

                    let present_image_info = vk::DescriptorImageInfo::builder()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(swapchain_image.view)
                        .build();

                    renderer.device.raw.cmd_bind_pipeline(
                        cb.raw,
                        vk::PipelineBindPoint::COMPUTE,
                        present_pipeline,
                    );

                    renderer
                        .device
                        .cmd_ext
                        .push_descriptor
                        .cmd_push_descriptor_set(
                            cb.raw,
                            vk::PipelineBindPoint::COMPUTE,
                            present_pipeline_layout,
                            0,
                            &[
                                /*vk::WriteDescriptorSet::builder()
                                    .dst_set(present_descriptor_set)
                                    .dst_binding(0)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(std::slice::from_ref(&image_info))
                                    .build(),
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(present_descriptor_set)
                                    .dst_binding(1)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(&[vk::DescriptorImageInfo::builder()
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(gui_texture_view)
                                        .build()])
                                    .build(),*/
                                vk::WriteDescriptorSet::builder()
                                    .dst_binding(2)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                    .image_info(std::slice::from_ref(&present_image_info))
                                    .build(),
                            ],
                        );

                    let output_size_pixels = (1280u32, 720u32); // TODO
                    let push_constants: (f32, f32) = (
                        1.0 / output_size_pixels.0 as f32,
                        1.0 / output_size_pixels.1 as f32,
                    );
                    renderer.device.raw.cmd_push_constants(
                        cb.raw,
                        present_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        std::slice::from_raw_parts(
                            &push_constants.0 as *const f32 as *const u8,
                            2 * 4,
                        ),
                    );
                    renderer.device.raw.cmd_dispatch(
                        cb.raw,
                        (output_size_pixels.0 + 7) / 8,
                        (output_size_pixels.1 + 7) / 8,
                        1,
                    );

                    record_image_barrier(
                        &renderer.device.raw,
                        cb.raw,
                        ImageBarrier::new(
                            swapchain_image.image,
                            vk_sync::AccessType::ComputeShaderWrite,
                            vk_sync::AccessType::Present,
                        ),
                    );

                    renderer.device.raw.end_command_buffer(cb.raw).unwrap();
                }

                let submit_info = vk::SubmitInfo::builder()
                    .wait_semaphores(std::slice::from_ref(&swapchain_image.acquire_semaphore))
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(std::slice::from_ref(&cb.raw));

                unsafe {
                    renderer
                        .device
                        .raw
                        .reset_fences(std::slice::from_ref(&cb.submit_done_fence))
                        .expect("reset_fences");

                    renderer
                        .device
                        .raw
                        .queue_submit(
                            renderer.device.universal_queue.raw,
                            &[submit_info.build()],
                            cb.submit_done_fence,
                        )
                        .expect("queue submit failed.");
                }

                renderer.swapchain.present_image(swapchain_image, &[]);
                renderer.device.finish_frame(current_frame);
            }
            _ => (),
        }
    })
}
