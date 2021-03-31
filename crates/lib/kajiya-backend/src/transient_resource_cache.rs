use crate::vulkan::{
    buffer::{Buffer, BufferDesc},
    image::{Image, ImageDesc},
};
use std::collections::HashMap;

#[derive(Default)]
pub struct TransientResourceCache {
    images: HashMap<ImageDesc, Vec<Image>>,
    buffers: HashMap<BufferDesc, Vec<Buffer>>,
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
            self.images.insert(image.desc, vec![image]);
        }
    }

    pub fn get_buffer(&mut self, desc: &BufferDesc) -> Option<Buffer> {
        if let Some(entry) = self.buffers.get_mut(desc) {
            entry.pop()
        } else {
            None
        }
    }

    pub fn insert_buffer(&mut self, buffer: Buffer) {
        if let Some(entry) = self.buffers.get_mut(&buffer.desc) {
            entry.push(buffer)
        } else {
            self.buffers.insert(buffer.desc, vec![buffer]);
        }
    }
}
