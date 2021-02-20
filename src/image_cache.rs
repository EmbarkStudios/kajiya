use std::{collections::HashMap, hash::Hash, path::PathBuf, sync::Arc};

use crate::asset::{
    image::{LoadImage, RawRgba8Image},
    mesh::{MeshMaterialMap, TexParams},
};
use slingshot::{ash::vk, Device, Image, ImageDesc, ImageSubResourceData};
use turbosloth::*;

pub enum ImageCacheResponse {
    Hit {
        id: usize,
    },
    Miss {
        id: usize,
        image: Lazy<RawRgba8Image>,
        params: TexParams,
    },
}
struct CachedImage {
    #[allow(dead_code)] // Stored to keep the lifetime
    lazy_handle: Lazy<RawRgba8Image>,
    id: usize,
}

pub struct ImageCache {
    lazy_cache: Arc<LazyCache>,
    loaded_images: HashMap<PathBuf, CachedImage>,
    placeholder_images: HashMap<[u8; 4], usize>,
    next_id: usize,
}

impl ImageCache {
    pub fn new(lazy_cache: Arc<LazyCache>) -> Self {
        Self {
            lazy_cache,
            loaded_images: Default::default(),
            placeholder_images: Default::default(),
            next_id: 0,
        }
    }

    /* pub fn load_mesh_map(&mut self, map: &MeshMaterialMap) -> anyhow::Result<ImageCacheResponse> {
        match map {
            MeshMaterialMap::Asset { path, params } => {
                if !self.loaded_images.contains_key(path) {
                    let image = LoadImage { path: path.clone() }.into_lazy();

                    let id = self.next_id;
                    self.next_id = self.next_id.checked_add(1).expect("Ran out of image IDs");

                    self.loaded_images.insert(
                        path.clone(),
                        CachedImage {
                            lazy_handle: image,
                            //image,
                            id,
                        },
                    );

                    Ok(ImageCacheResponse::Miss {
                        id,
                        image,
                        params: *params,
                    })
                } else {
                    Ok(ImageCacheResponse::Hit {
                        id: self.loaded_images[path].id,
                    })
                }
            }
            MeshMaterialMap::Placeholder(init_val) => {
                todo!()
                /* if !self.placeholder_images.contains_key(init_val) {
                    let image = Arc::new(RawRgba8Image {
                        data: init_val.to_vec(),
                        dimensions: [1, 1],
                    });

                    let id = self.next_id;
                    self.next_id = self.next_id.checked_add(1).expect("Ran out of image IDs");

                    self.placeholder_images.insert(*init_val, id);

                    Ok(ImageCacheResponse::Miss {
                        id,
                        image,
                        params: TexParams {
                            gamma: crate::asset::mesh::TexGamma::Linear,
                        },
                    })
                } else {
                    Ok(ImageCacheResponse::Hit {
                        id: self.placeholder_images[init_val],
                    })
                } */
            }
        }
    } */
}

#[derive(Clone)]
pub struct UploadGpuImage {
    pub image: Lazy<RawRgba8Image>,
    pub params: TexParams,
    pub device: Arc<Device>,
}

impl Hash for UploadGpuImage {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.image.hash(state);
        self.params.hash(state);
    }
}

#[async_trait]
impl LazyWorker for UploadGpuImage {
    type Output = anyhow::Result<Image>;

    async fn run(self, ctx: RunContext) -> Self::Output {
        let src = self.image.eval(&ctx).await?;

        let format = match self.params.gamma {
            crate::asset::mesh::TexGamma::Linear => vk::Format::R8G8B8A8_UNORM,
            crate::asset::mesh::TexGamma::Srgb => vk::Format::R8G8B8A8_SRGB,
        };

        self.device.create_image(
            ImageDesc::new_2d(format, src.dimensions).usage(vk::ImageUsageFlags::SAMPLED),
            Some(ImageSubResourceData {
                data: &src.data,
                row_pitch: src.dimensions[0] as usize * 4,
                slice_pitch: 0,
            }),
        )
    }
}
