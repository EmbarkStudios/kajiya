use std::sync::Arc;

use kajiya_backend::{vk_sync, vulkan::image::*};
use kajiya_rg as rg;

pub trait ComputeImageLut {
    fn create(&mut self, device: &kajiya_backend::Device) -> Image;
    fn compute(&mut self, rg: &mut rg::RenderGraph, img: &mut rg::Handle<Image>);
}

pub struct ImageLut {
    image: Arc<Image>,
    computer: Box<dyn ComputeImageLut>,
    computed: bool,
}

impl ImageLut {
    pub fn new(device: &kajiya_backend::Device, mut computer: Box<dyn ComputeImageLut>) -> Self {
        Self {
            image: Arc::new(computer.create(device)),
            computer,
            computed: false,
        }
    }

    pub fn compute_if_needed(&mut self, rg: &mut rg::RenderGraph) {
        if self.computed {
            return;
        }

        let mut rg_image = rg.import(self.image.clone(), vk_sync::AccessType::Nothing);

        self.computer.compute(rg, &mut rg_image);

        rg.export(
            rg_image,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        self.computed = true;
    }

    /// Note: contains garbage until `compute_if_needed` is called.
    pub fn backing_image(&self) -> Arc<Image> {
        self.image.clone()
    }
}

//pub fn clear_depth(rg: &mut RenderGraph, img: &mut Handle<Image>) {
