use std::{collections::HashMap, path::PathBuf, sync::Arc};

use crate::asset::{
    image::{LoadImage, RawRgba8Image},
    mesh::{MeshMaterialMap, TexParams},
};
use turbosloth::*;

pub enum ImageCacheResponse {
    Hit {
        id: usize,
    },
    Miss {
        id: usize,
        image: Arc<RawRgba8Image>,
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

    pub fn load_mesh_map(&mut self, map: &MeshMaterialMap) -> anyhow::Result<ImageCacheResponse> {
        match map {
            MeshMaterialMap::Asset { path, params } => {
                if !self.loaded_images.contains_key(path) {
                    let lazy_handle = LoadImage { path: path.clone() }.into_lazy();
                    let image = smol::block_on(lazy_handle.eval(&self.lazy_cache))?;

                    let id = self.next_id;
                    self.next_id = self.next_id.checked_add(1).expect("Ran out of image IDs");

                    self.loaded_images.insert(
                        path.clone(),
                        CachedImage {
                            lazy_handle,
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
                if !self.placeholder_images.contains_key(init_val) {
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
                }
            }
        }
    }
}
