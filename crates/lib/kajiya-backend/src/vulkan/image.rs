#![allow(clippy::let_and_return)]

use crate::BackendError;

use super::device::Device;
use ash::vk;
use derive_builder::Builder;
use gpu_allocator::{AllocationCreateDesc, MemoryLocation};
use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ImageType {
    Tex1d = 0,
    Tex1dArray = 1,
    Tex2d = 2,
    Tex2dArray = 3,
    Tex3d = 4,
    Cube = 5,
    CubeArray = 6,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ImageDesc {
    pub image_type: ImageType,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
    pub format: vk::Format,
    pub extent: [u32; 3],
    pub tiling: vk::ImageTiling,
    pub mip_levels: u16,
    pub array_elements: u32,
}

fn mip_count_1d(extent: u32) -> u16 {
    // floor(log2(extent)) + 1
    (32 - extent.leading_zeros()) as u16
}

impl ImageDesc {
    pub fn new(format: vk::Format, image_type: ImageType, extent: [u32; 3]) -> Self {
        Self {
            image_type,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::empty(),
            format,
            extent,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 1,
        }
    }

    pub fn new_1d(format: vk::Format, extent: u32) -> Self {
        Self::new(format, ImageType::Tex1d, [extent, 1, 1])
    }

    pub fn new_2d(format: vk::Format, extent: [u32; 2]) -> Self {
        let [width, height] = extent;
        Self::new(format, ImageType::Tex2d, [width, height, 1])
    }

    pub fn new_3d(format: vk::Format, extent: [u32; 3]) -> Self {
        Self::new(format, ImageType::Tex3d, extent)
    }

    pub fn new_cube(format: vk::Format, width: u32) -> Self {
        Self {
            image_type: ImageType::Cube,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
            format,
            extent: [width, width, 1],
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 6,
        }
    }

    pub fn image_type(mut self, image_type: ImageType) -> Self {
        self.image_type = image_type;
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn flags(mut self, flags: vk::ImageCreateFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = format;
        self
    }

    pub fn extent(mut self, extent: [u32; 3]) -> Self {
        self.extent = extent;
        self
    }

    pub fn tiling(mut self, tiling: vk::ImageTiling) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn mip_levels(mut self, mip_levels: u16) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    pub fn all_mip_levels(mut self) -> Self {
        self.mip_levels = mip_count_1d(self.extent[0])
            .max(mip_count_1d(self.extent[1]).max(mip_count_1d(self.extent[2])));
        self
    }

    pub fn array_elements(mut self, array_elements: u32) -> Self {
        self.array_elements = array_elements;
        self
    }

    pub fn div_up_extent(mut self, div_extent: [u32; 3]) -> Self {
        for (extent, &div_extent) in self.extent.iter_mut().zip(&div_extent) {
            *extent = ((*extent + div_extent - 1) / div_extent).max(1);
        }
        self
    }

    pub fn div_extent(mut self, div_extent: [u32; 3]) -> Self {
        for (extent, &div_extent) in self.extent.iter_mut().zip(&div_extent) {
            *extent = (*extent / div_extent).max(1);
        }
        self
    }

    pub fn half_res(self) -> Self {
        self.div_up_extent([2, 2, 2])
    }

    pub fn extent_inv_extent_2d(&self) -> [f32; 4] {
        [
            self.extent[0] as f32,
            self.extent[1] as f32,
            1.0 / self.extent[0] as f32,
            1.0 / self.extent[1] as f32,
        ]
    }

    pub fn extent_2d(&self) -> [u32; 2] {
        [self.extent[0], self.extent[1]]
    }
}

pub struct ImageSubResourceData<'a> {
    pub data: &'a [u8],
    pub row_pitch: usize,
    pub slice_pitch: usize,
}

pub struct Image {
    pub raw: vk::Image,
    pub desc: ImageDesc,
    pub views: Mutex<HashMap<ImageViewDesc, vk::ImageView>>,
    //allocation: gpu_allocator::SubAllocation,
}
unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    pub fn view(
        &self,
        device: &Device,
        desc: &ImageViewDesc,
    ) -> Result<vk::ImageView, BackendError> {
        let mut views = self.views.lock();

        if let Some(entry) = views.get(desc) {
            Ok(*entry)
        } else {
            let view = device.create_image_view(*desc, &self.desc, self.raw)?;
            Ok(*views.entry(*desc).or_insert(view))
        }
    }

    pub fn view_desc(&self, desc: &ImageViewDesc) -> vk::ImageViewCreateInfo {
        Self::view_desc_impl(*desc, &self.desc)
    }

    fn view_desc_impl(desc: ImageViewDesc, image_desc: &ImageDesc) -> vk::ImageViewCreateInfo {
        vk::ImageViewCreateInfo::builder()
            .format(desc.format.unwrap_or(image_desc.format))
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .view_type(
                desc.view_type
                    .unwrap_or_else(|| convert_image_type_to_view_type(image_desc.image_type)),
            )
            // TODO
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: desc.aspect_mask,
                base_mip_level: desc.base_mip_level,
                level_count: desc.level_count.unwrap_or(image_desc.mip_levels as u32),
                base_array_layer: 0,
                layer_count: match image_desc.image_type {
                    ImageType::Cube | ImageType::CubeArray => 6,
                    _ => 1,
                },
            })
            .build()
    }
}

#[derive(Clone, Copy, Builder, Eq, PartialEq, Hash)]
#[builder(pattern = "owned", derive(Clone))]
pub struct ImageViewDesc {
    #[builder(setter(strip_option), default)]
    pub view_type: Option<vk::ImageViewType>,
    #[builder(setter(strip_option), default)]
    pub format: Option<vk::Format>,
    #[builder(default = "vk::ImageAspectFlags::COLOR")]
    pub aspect_mask: vk::ImageAspectFlags,
    #[builder(default = "0")]
    pub base_mip_level: u32,
    #[builder(default = "None")]
    pub level_count: Option<u32>,
    // TODO
}

impl ImageViewDesc {
    pub fn builder() -> ImageViewDescBuilder {
        Default::default()
    }
}

impl Default for ImageViewDesc {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl Device {
    pub fn create_image(
        &self,
        desc: ImageDesc,
        initial_data: Vec<ImageSubResourceData>,
    ) -> Result<Image, BackendError> {
        log::info!("Creating an image: {:?}", desc);

        let create_info = get_image_create_info(&desc, !initial_data.is_empty());

        /*let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _allocation_info) = self
            .global_allocator
            .create_image(&create_info, &allocation_info)?;*/

        let image = unsafe {
            self.raw
                .create_image(&create_info, None)
                .expect("create_image")
        };
        let requirements = unsafe { self.raw.get_image_memory_requirements(image) };

        let allocation = self
            .global_allocator
            .lock()
            .allocate(&AllocationCreateDesc {
                name: "image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
            })
            .map_err(|err| BackendError::Allocation {
                inner: err,
                name: "GpuOnly image".into(),
            })?;

        // Bind memory to the image
        unsafe {
            self.raw
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .expect("bind_image_memory")
        };

        if !initial_data.is_empty() {
            let total_initial_data_bytes = initial_data.iter().map(|d| d.data.len()).sum();

            let block_bytes: usize = match desc.format {
                vk::Format::R8G8B8A8_UNORM => 1,
                vk::Format::R8G8B8A8_SRGB => 1,
                vk::Format::R32G32B32A32_SFLOAT => 1,
                vk::Format::R16G16B16A16_SFLOAT => 1,
                vk::Format::BC1_RGB_UNORM_BLOCK => 8,
                vk::Format::BC1_RGB_SRGB_BLOCK => 8,
                vk::Format::BC3_UNORM_BLOCK => 16,
                vk::Format::BC3_SRGB_BLOCK => 16,
                vk::Format::BC5_UNORM_BLOCK => 16,
                vk::Format::BC5_SNORM_BLOCK => 16,
                vk::Format::BC7_UNORM_BLOCK => 16,
                vk::Format::BC7_SRGB_BLOCK => 16,
                _ => todo!("{:?}", desc.format),
            };

            let mut image_buffer = self.create_buffer(
                super::buffer::BufferDesc::new_cpu_to_gpu(
                    total_initial_data_bytes,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                ),
                "Image initial data buffer",
                None,
            )?;

            let mapped_slice_mut = image_buffer.allocation.mapped_slice_mut().unwrap();
            let mut offset = 0;

            let buffer_copy_regions = initial_data
                .into_iter()
                .enumerate()
                .map(|(level, sub)| {
                    mapped_slice_mut[offset..offset + sub.data.len()].copy_from_slice(sub.data);
                    assert_eq!(offset % block_bytes, 0);

                    let region = vk::BufferImageCopy::builder()
                        .buffer_offset(offset as _)
                        .image_subresource(
                            vk::ImageSubresourceLayers::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .mip_level(level as _)
                                .build(),
                        )
                        .image_extent(vk::Extent3D {
                            width: (desc.extent[0] >> level).max(1),
                            height: (desc.extent[1] >> level).max(1),
                            depth: (desc.extent[2] >> level).max(1),
                        });

                    offset += sub.data.len();
                    let region = region.build();

                    //dbg!(region);
                    //dbg!(total_initial_data_bytes);

                    region
                })
                .collect::<Vec<_>>();

            // println!("regions: {:#?}", buffer_copy_regions);

            let copy_result = self.with_setup_cb(|cb| unsafe {
                super::barrier::record_image_barrier(
                    self,
                    cb,
                    super::barrier::ImageBarrier::new(
                        image,
                        vk_sync::AccessType::Nothing,
                        vk_sync::AccessType::TransferWrite,
                        vk::ImageAspectFlags::COLOR,
                    )
                    .with_discard(true),
                );

                self.raw.cmd_copy_buffer_to_image(
                    cb,
                    image_buffer.raw,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &buffer_copy_regions,
                );

                super::barrier::record_image_barrier(
                    self,
                    cb,
                    super::barrier::ImageBarrier::new(
                        image,
                        vk_sync::AccessType::TransferWrite,
                        vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                        vk::ImageAspectFlags::COLOR,
                    ),
                )
            });

            self.immediate_destroy_buffer(image_buffer);

            copy_result?;
        }

        /*        let handle = self.storage.insert(Image {
            raw: image,
            allocation,
        });

        ImageHandle(handle)*/
        Ok(Image {
            raw: image,
            //allocation,
            desc,
            views: Default::default(),
        })
    }

    fn create_image_view(
        &self,
        desc: ImageViewDesc,
        image_desc: &ImageDesc,
        image_raw: vk::Image,
    ) -> Result<vk::ImageView, BackendError> {
        if image_desc.format == vk::Format::D32_SFLOAT
            && !desc.aspect_mask.contains(vk::ImageAspectFlags::DEPTH)
        {
            return Err(BackendError::ResourceAccess {
                info: "Depth-only resource used without the vk::ImageAspectFlags::DEPTH flag"
                    .to_owned(),
            });
        }

        let create_info = vk::ImageViewCreateInfo {
            image: image_raw,
            ..Image::view_desc_impl(desc, image_desc)
        };

        Ok(unsafe { self.raw.create_image_view(&create_info, None)? })
    }

    /*pub fn get(&self, handle: ImageHandle) -> &Image {
        self.storage.get(handle.0)
    }

    pub fn maintain(&mut self) {
        self.storage.maintain()
    }*/
}

pub fn convert_image_type_to_view_type(image_type: ImageType) -> vk::ImageViewType {
    match image_type {
        ImageType::Tex1d => vk::ImageViewType::TYPE_1D,
        ImageType::Tex1dArray => vk::ImageViewType::TYPE_1D_ARRAY,
        ImageType::Tex2d => vk::ImageViewType::TYPE_2D,
        ImageType::Tex2dArray => vk::ImageViewType::TYPE_2D_ARRAY,
        ImageType::Tex3d => vk::ImageViewType::TYPE_3D,
        ImageType::Cube => vk::ImageViewType::CUBE,
        ImageType::CubeArray => vk::ImageViewType::CUBE_ARRAY,
    }
}

pub fn get_image_create_info(desc: &ImageDesc, initial_data: bool) -> vk::ImageCreateInfo {
    let (image_type, image_extent, image_layers) = match desc.image_type {
        ImageType::Tex1d => (
            vk::ImageType::TYPE_1D,
            vk::Extent3D {
                width: desc.extent[0],
                height: 1,
                depth: 1,
            },
            1,
        ),
        ImageType::Tex1dArray => (
            vk::ImageType::TYPE_1D,
            vk::Extent3D {
                width: desc.extent[0],
                height: 1,
                depth: 1,
            },
            desc.array_elements,
        ),
        ImageType::Tex2d => (
            vk::ImageType::TYPE_2D,
            vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            },
            1,
        ),
        ImageType::Tex2dArray => (
            vk::ImageType::TYPE_2D,
            vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            },
            desc.array_elements,
        ),
        ImageType::Tex3d => (
            vk::ImageType::TYPE_3D,
            vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: desc.extent[2],
            },
            1,
        ),
        ImageType::Cube => (
            vk::ImageType::TYPE_2D,
            vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            },
            6,
        ),
        ImageType::CubeArray => (
            vk::ImageType::TYPE_2D,
            vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            },
            6 * desc.array_elements,
        ),
    };

    let mut image_usage = desc.usage;

    if initial_data {
        image_usage |= vk::ImageUsageFlags::TRANSFER_DST;
    }

    vk::ImageCreateInfo {
        flags: desc.flags,
        image_type,
        format: desc.format,
        extent: image_extent,
        mip_levels: desc.mip_levels as u32,
        array_layers: image_layers,
        samples: vk::SampleCountFlags::TYPE_1, // TODO: desc.sample_count
        tiling: desc.tiling,
        usage: image_usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: match initial_data {
            true => vk::ImageLayout::PREINITIALIZED,
            false => vk::ImageLayout::UNDEFINED,
        },
        ..Default::default()
    }
}
