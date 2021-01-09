use super::device::Device;
use anyhow::Result;
use ash::{util::Align, version::DeviceV1_0, vk};
use derive_builder::Builder;
use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[allow(dead_code)]
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
    pub flags: vk::ImageCreateFlags, // TODO: CUBE_COMPATIBLE
    pub format: vk::Format,
    pub extent: [u32; 3],
    pub tiling: vk::ImageTiling,
    pub mip_levels: u16,
    pub array_elements: u32,
}

#[allow(dead_code)]
impl ImageDesc {
    pub fn new(format: vk::Format, image_type: ImageType, extent: [u32; 3]) -> Self {
        Self {
            image_type,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::MUTABLE_FORMAT, //TODO: CUBE_COMPATIBLE
            format,
            extent,
            tiling: vk::ImageTiling::OPTIMAL,
            mip_levels: 1,
            array_elements: 1,
        }
    }

    pub fn new_2d(format: vk::Format, extent: [u32; 2]) -> Self {
        let [width, height] = extent;
        Self::new(format, ImageType::Tex2d, [width, height, 1])
    }

    pub fn new_3d(format: vk::Format, extent: [u32; 3]) -> Self {
        Self::new(format, ImageType::Tex3d, extent)
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

    pub fn array_elements(mut self, array_elements: u32) -> Self {
        self.array_elements = array_elements;
        self
    }
}

pub struct ImageSubResourceData<'a> {
    pub data: &'a [u8],
    pub row_pitch: usize,
    pub slice_pitch: usize,
}

#[allow(dead_code)]
pub struct Image {
    pub raw: vk::Image,
    pub desc: ImageDesc,
    pub views: Mutex<HashMap<ImageViewDesc, vk::ImageView>>,
    allocation: vk_mem::Allocation,
}

impl Image {
    pub fn view(&self, device: &Device, desc: &ImageViewDesc) -> vk::ImageView {
        let mut views = self.views.lock();

        if let Some(entry) = views.get(desc) {
            *entry
        } else {
            *views.entry(desc.clone()).or_insert_with(|| {
                device
                    .create_image_view(desc.clone(), &self.desc, self.raw)
                    .unwrap()
            })
        }
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
    // TODO
}

impl ImageViewDesc {
    #[allow(dead_code)]
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
        initial_data: Option<ImageSubResourceData>,
    ) -> Result<Image> {
        log::info!("Creating an image: {:?}", desc);

        let desc = desc.into();
        let create_info = get_image_create_info(&desc, initial_data.is_some());

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _allocation_info) = self
            .global_allocator
            .create_image(&create_info, &allocation_info)?;

        if let Some(initial_data) = initial_data {
            let image_buffer_info = vk::BufferCreateInfo {
                size: (std::mem::size_of::<u8>() * initial_data.data.len()) as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer_mem_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::CpuToGpu,
                ..Default::default()
            };

            let (image_buffer, buffer_allocation, buffer_allocation_info) = self
                .global_allocator
                .create_buffer(&image_buffer_info, &buffer_mem_info)
                .expect("vma::create_buffer");

            unsafe {
                let image_ptr = self
                    .global_allocator
                    .map_memory(&buffer_allocation)
                    .expect("mapping an image upload buffer failed")
                    as *mut std::ffi::c_void;

                let mut image_slice = Align::new(
                    image_ptr,
                    std::mem::align_of::<u8>() as u64,
                    buffer_allocation_info.get_size() as u64,
                );

                image_slice.copy_from_slice(initial_data.data);

                self.global_allocator
                    .unmap_memory(&buffer_allocation)
                    .expect("unmap_memory");
            }

            self.with_setup_cb(|cb| unsafe {
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

                let buffer_copy_regions = vk::BufferImageCopy::builder()
                    .image_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .build(),
                    )
                    .image_extent(vk::Extent3D {
                        width: desc.extent[0],
                        height: desc.extent[1],
                        depth: desc.extent[2],
                    });

                self.raw.cmd_copy_buffer_to_image(
                    cb,
                    image_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_copy_regions.build()],
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
        }

        /*        let handle = self.storage.insert(Image {
            raw: image,
            allocation,
        });

        ImageHandle(handle)*/
        Ok(Image {
            raw: image,
            allocation,
            desc,
            views: Default::default(),
        })
    }

    fn create_image_view(
        &self,
        desc: ImageViewDesc,
        image_desc: &ImageDesc,
        image_raw: vk::Image,
    ) -> Result<vk::ImageView> {
        let create_info = vk::ImageViewCreateInfo::builder()
            .format(desc.format.unwrap_or(image_desc.format))
            .image(image_raw)
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
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: match image_desc.image_type {
                    ImageType::Cube | ImageType::CubeArray => 6,
                    _ => 1,
                },
            })
            .build();

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
                depth: desc.extent[2] as u32,
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
        array_layers: image_layers as u32,
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
