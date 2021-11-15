use super::{device::Device, surface::Surface};
use anyhow::Result;
use ash::{extensions::khr, vk};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;

#[derive(Clone, Copy, Default)]
pub struct SwapchainDesc {
    pub format: vk::SurfaceFormatKHR,
    pub dims: vk::Extent2D,
    pub vsync: bool,
}

pub struct Swapchain {
    pub(crate) fns: khr::Swapchain,
    pub(crate) raw: vk::SwapchainKHR,
    pub desc: SwapchainDesc,
    pub images: Vec<Arc<crate::Image>>,
    pub acquire_semaphores: Vec<vk::Semaphore>,

    // TODO: move out of swapchain, make a single semaphore
    pub rendering_finished_semaphores: Vec<vk::Semaphore>,
    pub next_semaphore: usize,

    // Keep a reference in order not to drop after the device
    #[allow(dead_code)]
    pub(crate) device: Arc<Device>,

    // Ditto
    #[allow(dead_code)]
    surface: Arc<Surface>,
}

pub struct SwapchainImage {
    pub image: Arc<crate::Image>,
    pub image_index: u32,
    pub acquire_semaphore: vk::Semaphore,
    pub rendering_finished_semaphore: vk::Semaphore,
}

pub enum SwapchainAcquireImageErr {
    RecreateFramebuffer,
}

impl Swapchain {
    pub fn enumerate_surface_formats(
        device: &Arc<Device>,
        surface: &Surface,
    ) -> Result<Vec<vk::SurfaceFormatKHR>> {
        unsafe {
            Ok(surface
                .fns
                .get_physical_device_surface_formats(device.pdevice.raw, surface.raw)?)
        }
    }

    pub fn new(device: &Arc<Device>, surface: &Arc<Surface>, desc: SwapchainDesc) -> Result<Self> {
        let surface_capabilities = unsafe {
            surface
                .fns
                .get_physical_device_surface_capabilities(device.pdevice.raw, surface.raw)
        }?;

        // Triple-buffer so that acquiring an image doesn't stall for >16.6ms at 60Hz on AMD
        // when frames take >16.6ms to render. Also allows MAILBOX to work.
        let mut desired_image_count = 3.max(surface_capabilities.min_image_count);

        if surface_capabilities.max_image_count != 0 {
            desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
        }

        log::info!("Swapchain image count: {}", desired_image_count);

        //dbg!(&surface_capabilities);
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => desc.dims,
            _ => surface_capabilities.current_extent,
        };

        if 0 == surface_resolution.width || 0 == surface_resolution.height {
            anyhow::bail!("Swapchain resolution cannot be zero");
        }

        let present_mode_preference = if desc.vsync {
            vec![vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO]
        } else {
            vec![vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
        };

        let present_modes = unsafe {
            surface
                .fns
                .get_physical_device_surface_present_modes(device.pdevice.raw, surface.raw)
        }?;

        let present_mode = present_mode_preference
            .into_iter()
            .find(|mode| present_modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO);
        log::info!("Presentation mode: {:?}", present_mode);

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.raw)
            .min_image_count(desired_image_count)
            .image_color_space(desc.format.color_space)
            .image_format(desc.format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::STORAGE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .build();

        let fns = khr::Swapchain::new(&device.instance.raw, &device.raw);
        let swapchain = unsafe { fns.create_swapchain(&swapchain_create_info, None) }.unwrap();

        let vk_images = unsafe { fns.get_swapchain_images(swapchain) }.unwrap();
        /*let image_views = images
        .iter()
        .map(|image| unsafe {
            device
                .raw
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        image: *image,
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
                .expect("create_image_view")
        })
        .collect();*/

        let images: Vec<Arc<crate::Image>> = vk_images
            .into_iter()
            .map(|vk_image| {
                Arc::new(crate::Image {
                    raw: vk_image,
                    desc: crate::ImageDesc {
                        image_type: crate::ImageType::Tex2d,
                        usage: vk::ImageUsageFlags::STORAGE,
                        flags: vk::ImageCreateFlags::empty(),
                        format: vk::Format::B8G8R8A8_UNORM,
                        extent: [desc.dims.width, desc.dims.height, 0],
                        tiling: vk::ImageTiling::OPTIMAL,
                        mip_levels: 1,
                        array_elements: 1,
                    },
                    views: Default::default(),
                })
            })
            .collect();

        assert_eq!(desired_image_count, images.len() as u32);

        let acquire_semaphores = (0..images.len())
            .map(|_| {
                unsafe {
                    device
                        .raw
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                }
                .unwrap()
            })
            .collect();

        let rendering_finished_semaphores = (0..images.len())
            .map(|_| {
                unsafe {
                    device
                        .raw
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                }
                .unwrap()
            })
            .collect();

        Ok(Swapchain {
            fns,
            raw: swapchain,
            desc,
            images,
            acquire_semaphores,
            rendering_finished_semaphores,
            next_semaphore: 0,
            device: device.clone(),
            surface: surface.clone(),
        })
    }

    pub fn extent(&self) -> [u32; 2] {
        [self.desc.dims.width, self.desc.dims.height]
    }

    pub fn acquire_next_image(
        &mut self,
    ) -> std::result::Result<SwapchainImage, SwapchainAcquireImageErr> {
        puffin::profile_function!();

        let acquire_semaphore = self.acquire_semaphores[self.next_semaphore];
        let rendering_finished_semaphore = self.rendering_finished_semaphores[self.next_semaphore];

        let present_index = unsafe {
            self.fns.acquire_next_image(
                self.raw,
                std::u64::MAX,
                acquire_semaphore,
                vk::Fence::null(),
            )
        }
        .map(|(val, _)| val as usize);

        match present_index {
            Ok(present_index) => {
                assert_eq!(present_index, self.next_semaphore);

                self.next_semaphore = (self.next_semaphore + 1) % self.images.len();
                Ok(SwapchainImage {
                    image: self.images[present_index].clone(),
                    image_index: present_index as u32,
                    acquire_semaphore,
                    rendering_finished_semaphore,
                })
            }
            Err(err)
                if err == vk::Result::ERROR_OUT_OF_DATE_KHR
                    || err == vk::Result::SUBOPTIMAL_KHR =>
            {
                Err(SwapchainAcquireImageErr::RecreateFramebuffer)
            }
            err => {
                panic!("Could not acquire swapchain image: {:?}", err);
            }
        }
    }

    pub fn present_image(&self, image: SwapchainImage) {
        puffin::profile_function!();

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(&image.rendering_finished_semaphore))
            .swapchains(std::slice::from_ref(&self.raw))
            .image_indices(std::slice::from_ref(&image.image_index));

        unsafe {
            match self
                .fns
                .queue_present(self.device.universal_queue.raw, &present_info)
            {
                Ok(_) => (),
                Err(err)
                    if err == vk::Result::ERROR_OUT_OF_DATE_KHR
                        || err == vk::Result::SUBOPTIMAL_KHR =>
                {
                    // Handled in the next frame
                }
                err => {
                    panic!("Could not present image: {:?}", err);
                }
            }
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.fns.destroy_swapchain(self.raw, None);
        }
    }
}
