mod device;
mod instance;
mod logging;
mod physical_device;
mod presentation;
mod surface;
mod swapchain;

use ash::extensions::khr::PushDescriptor as _;
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
        .graphics_debugging(true)
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

    let (present_pipeline_layout, present_pipeline) =
        presentation::create_present_descriptor_set_and_pipeline(&*device);

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
                let swapchain_image = swapchain
                    .acquire_next_image()
                    .ok()
                    .expect("swapchain image");

                // TODO
                let command_buffer = device.frames[0].command_buffer.raw;

                unsafe {
                    device
                        .raw
                        .begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                        )
                        .unwrap();

                    record_image_barrier(
                        &device.raw,
                        command_buffer,
                        ImageBarrier::new(
                            swapchain_image.image,
                            vk_sync::AccessType::Present,
                            vk_sync::AccessType::ComputeShaderWrite,
                        )
                        .with_discard(true),
                    );

                    let present_image_view = device
                        .raw
                        .create_image_view(
                            &vk::ImageViewCreateInfo {
                                image: swapchain_image.image,
                                view_type: vk::ImageViewType::TYPE_2D,
                                format: vk::Format::B8G8R8A8_UNORM,
                                subresource_range: vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    level_count: 1,
                                    layer_count: 1,
                                    ..Default::default()
                                },
                                components: vk::ComponentMapping {
                                    r: vk::ComponentSwizzle::R,
                                    g: vk::ComponentSwizzle::G,
                                    b: vk::ComponentSwizzle::B,
                                    a: vk::ComponentSwizzle::A,
                                },
                                ..Default::default()
                            },
                            None,
                        )
                        .expect("create_image_view");

                    let present_image_info = vk::DescriptorImageInfo::builder()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(present_image_view)
                        .build();

                    device.cmd_ext.push_descriptor.cmd_push_descriptor_set(
                        command_buffer,
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

                    device.raw.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        present_pipeline,
                    );

                    let output_size_pixels = (1280u32, 720u32); // TODO
                    let push_constants: (f32, f32) = (
                        1.0 / output_size_pixels.0 as f32,
                        1.0 / output_size_pixels.1 as f32,
                    );
                    device.raw.cmd_push_constants(
                        command_buffer,
                        present_pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        std::slice::from_raw_parts(
                            &push_constants.0 as *const f32 as *const u8,
                            2 * 4,
                        ),
                    );
                    device.raw.cmd_dispatch(
                        command_buffer,
                        (output_size_pixels.0 + 7) / 8,
                        (output_size_pixels.1 + 7) / 8,
                        1,
                    );

                    record_image_barrier(
                        &device.raw,
                        command_buffer,
                        ImageBarrier::new(
                            swapchain_image.image,
                            vk_sync::AccessType::ComputeShaderWrite,
                            vk_sync::AccessType::Present,
                        ),
                    );

                    device.raw.end_command_buffer(command_buffer).unwrap();
                }

                let submit_info = vk::SubmitInfo::builder()
                    .wait_semaphores(std::slice::from_ref(&swapchain_image.acquire_semaphore))
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(std::slice::from_ref(&command_buffer));

                unsafe {
                    device
                        .raw
                        .queue_submit(
                            device.universal_queue.raw,
                            &[submit_info.build()],
                            vk::Fence::default(),
                        )
                        .expect("queue submit failed.");
                }

                swapchain.present_image(swapchain_image, &[]);

                std::process::exit(0);
            }
            _ => (),
        }
    })
}

pub struct ImageBarrier {
    image: vk::Image,
    prev_access: vk_sync::AccessType,
    next_access: vk_sync::AccessType,
    discard: bool,
}

pub fn record_image_barrier(device: &ash::Device, cb: vk::CommandBuffer, barrier: ImageBarrier) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    vk_sync::cmd::pipeline_barrier(
        device.fp_v1_0(),
        cb,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[barrier.prev_access],
            next_accesses: &[barrier.next_access],
            previous_layout: vk_sync::ImageLayout::Optimal,
            next_layout: vk_sync::ImageLayout::Optimal,
            discard_contents: barrier.discard,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: barrier.image,
            range,
        }],
    );
}

impl ImageBarrier {
    pub fn new(
        image: vk::Image,
        prev_access: vk_sync::AccessType,
        next_access: vk_sync::AccessType,
    ) -> Self {
        Self {
            image,
            prev_access,
            next_access,
            discard: false,
        }
    }

    pub fn with_discard(mut self, discard: bool) -> Self {
        self.discard = discard;
        self
    }
}
