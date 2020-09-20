use parking_lot::Mutex;

use crate::backend::image::{Image, ImageView, ImageViewDesc};
use std::{collections::HashMap, hash::Hash, sync::Arc, sync::Weak};

pub(crate) struct ImageViewCacheKey {
    pub(crate) image: Weak<Image>,
    pub(crate) view_desc: ImageViewDesc,
}
impl PartialEq for ImageViewCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.image.as_ptr() == other.image.as_ptr()
    }
}
impl Eq for ImageViewCacheKey {}
impl Hash for ImageViewCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.image.as_ptr() as usize).hash(state);
        self.view_desc.hash(state);
    }
}

#[derive(Default)]
pub struct ViewCache {
    pub(crate) image_views: Mutex<HashMap<ImageViewCacheKey, Arc<ImageView>>>,
}
