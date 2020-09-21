use crate::backend::image::{Image, ImageDesc};
use std::collections::HashMap;

#[derive(Default)]
pub struct TransientResourceCache {
    images: HashMap<ImageDesc, Vec<Image>>,
}

impl TransientResourceCache {
    pub fn get_image(&mut self, desc: &ImageDesc) -> Option<Image> {
        if let Some(entry) = self.images.get_mut(desc) {
            entry.pop()
        } else {
            None
        }
    }

    pub fn insert_image(&mut self, image: Image) {
        if let Some(entry) = self.images.get_mut(&image.desc) {
            entry.push(image)
        } else {
            self.images.insert(image.desc.clone(), vec![image]);
        }
    }
}
