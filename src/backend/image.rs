use super::device::Device;
use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use derive_builder::Builder;
use std::sync::Arc;

#[derive(Clone, Copy)]
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

#[derive(Builder, Clone, Copy)]
#[builder(pattern = "owned")]
pub struct ImageDesc {
    pub image_type: ImageType,
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub extent: [u32; 3],
    #[builder(default = "vk::ImageTiling::OPTIMAL")]
    pub tiling: vk::ImageTiling,
    #[builder(default = "1")]
    pub mip_levels: u16,
    #[builder(default = "1")]
    pub array_elements: u32,
}

#[allow(dead_code)]
impl ImageDesc {
    pub fn new() -> ImageDescBuilder {
        ImageDescBuilder::default()
    }

    pub fn new_2d(extent: [u32; 2]) -> ImageDescBuilder {
        let [width, height] = extent;
        ImageDescBuilder::default()
            .extent([width, height, 1])
            .image_type(ImageType::Tex2d)
    }

    pub fn new_3d(extent: [u32; 3]) -> ImageDescBuilder {
        ImageDescBuilder::default()
            .extent(extent)
            .image_type(ImageType::Tex3d)
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
    allocation: vk_mem::Allocation,
}

#[derive(Clone, Builder)]
pub struct ImageViewDesc {
    pub image: Arc<Image>,
    #[builder(setter(strip_option), default)]
    pub view_type: Option<vk::ImageViewType>,
    #[builder(setter(strip_option), default)]
    pub format: Option<vk::Format>,
    // TODO
}

impl ImageViewDesc {
    pub fn builder() -> ImageViewDescBuilder {
        Default::default()
    }
}

pub struct ImageView {
    pub raw: vk::ImageView,
    desc: ImageViewDesc,
}

impl Device {
    pub fn create_image(
        &self,
        desc: ImageDesc,
        initial_data: Option<ImageSubResourceData>,
    ) -> Result<Arc<Image>> {
        let desc = desc.into();
        let create_info = get_image_create_info(&desc, initial_data.is_some());

        if initial_data.is_some() {
            todo!();
        }

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _allocation_info) = self
            .global_allocator
            .create_image(&create_info, &allocation_info)?;

        /*        let handle = self.storage.insert(Image {
            raw: image,
            allocation,
        });

        ImageHandle(handle)*/
        Ok(Arc::new(Image {
            raw: image,
            allocation,
            desc,
        }))
    }

    pub fn create_image_view(&self, desc: ImageViewDesc) -> Result<ImageView> {
        let image = &*desc.image;
        let create_info = vk::ImageViewCreateInfo::builder()
            .format(desc.format.unwrap_or(image.desc.format))
            .image(image.raw)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::A,
            })
            .view_type(
                desc.view_type
                    .unwrap_or_else(|| convert_image_type_to_view_type(image.desc.image_type)),
            )
            // TODO
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: match image.desc.image_type {
                    ImageType::Cube | ImageType::CubeArray => 6,
                    _ => 1,
                },
            })
            .build();

        let raw = unsafe { self.raw.create_image_view(&create_info, None)? };
        Ok(ImageView { raw, desc })
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
        flags: match desc.image_type {
            ImageType::Cube => vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ImageType::CubeArray => vk::ImageCreateFlags::CUBE_COMPATIBLE,
            _ => vk::ImageCreateFlags::empty(), // ImageCreateFlags::CREATE_MUTABLE_FORMAT
        },
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
