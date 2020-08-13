use super::{device::Device, resource_storage::*};
use ash::vk;
use derive_builder::Builder;

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

#[derive(Builder)]
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
    allocation: vk_mem::Allocation,
}

pub struct ImageHandle(ResourceHandle);

#[derive(Default)]
pub struct ImageStorage {
    storage: ResourceStorage<Image>,
}

impl ImageStorage {
    pub fn create(
        &self,
        device: &Device,
        desc: &ImageDesc,
        initial_data: Option<ImageSubResourceData>,
    ) -> ImageHandle {
        let create_info = get_image_create_info(&desc, initial_data.is_some());

        if initial_data.is_some() {
            todo!();
        }

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _allocation_info) = device
            .global_allocator
            .create_image(&create_info, &allocation_info)
            .unwrap();

        let handle = self.storage.insert(Image {
            raw: image,
            allocation,
        });

        ImageHandle(handle)
    }

    pub fn get(&self, handle: ImageHandle) -> &Image {
        self.storage.get(handle.0)
    }

    pub fn maintain(&mut self) {
        self.storage.maintain()
    }
}

pub fn get_image_create_info(desc: &ImageDesc, initial_data: bool) -> ash::vk::ImageCreateInfo {
    let (image_type, image_extent, image_layers) = match desc.image_type {
        ImageType::Tex1d => (
            ash::vk::ImageType::TYPE_1D,
            ash::vk::Extent3D {
                width: desc.extent[0],
                height: 1,
                depth: 1,
            },
            1,
        ),
        ImageType::Tex1dArray => (
            ash::vk::ImageType::TYPE_1D,
            ash::vk::Extent3D {
                width: desc.extent[0],
                height: 1,
                depth: 1,
            },
            desc.array_elements,
        ),
        ImageType::Tex2d => (
            ash::vk::ImageType::TYPE_2D,
            ash::vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            },
            1,
        ),
        ImageType::Tex2dArray => (
            ash::vk::ImageType::TYPE_2D,
            ash::vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            },
            desc.array_elements,
        ),
        ImageType::Tex3d => (
            ash::vk::ImageType::TYPE_3D,
            ash::vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: desc.extent[2] as u32,
            },
            1,
        ),
        ImageType::Cube => (
            ash::vk::ImageType::TYPE_2D,
            ash::vk::Extent3D {
                width: desc.extent[0],
                height: desc.extent[1],
                depth: 1,
            },
            6,
        ),
        ImageType::CubeArray => (
            ash::vk::ImageType::TYPE_2D,
            ash::vk::Extent3D {
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

    ash::vk::ImageCreateInfo {
        flags: match desc.image_type {
            ImageType::Cube => ash::vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ImageType::CubeArray => ash::vk::ImageCreateFlags::CUBE_COMPATIBLE,
            _ => ash::vk::ImageCreateFlags::empty(), // ImageCreateFlags::CREATE_MUTABLE_FORMAT
        },
        image_type,
        format: desc.format,
        extent: image_extent,
        mip_levels: desc.mip_levels as u32,
        array_layers: image_layers as u32,
        samples: ash::vk::SampleCountFlags::TYPE_1, // TODO: desc.sample_count
        tiling: desc.tiling,
        usage: image_usage,
        sharing_mode: ash::vk::SharingMode::EXCLUSIVE,
        initial_layout: match initial_data {
            true => ash::vk::ImageLayout::PREINITIALIZED,
            false => ash::vk::ImageLayout::UNDEFINED,
        },
        ..Default::default()
    }
}
