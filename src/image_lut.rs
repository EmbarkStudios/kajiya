use std::sync::Arc;

use crate::{
    asset::{
        image::RawRgba8Image,
        mesh::{PackedTriangleMesh, PackedVertex},
    },
    backend::{self, image::*, shader::*, RenderBackend},
    dynamic_constants::DynamicConstants,
    render_passes::{RasterMeshesData, UploadedTriMesh},
    renderer::*,
    rg,
    rg::RetiredRenderGraph,
    viewport::ViewConstants,
    FrameState,
};
use backend::buffer::{Buffer, BufferDesc};
use glam::Vec2;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use parking_lot::Mutex;
use slingshot::{
    ash::{version::DeviceV1_0, vk},
    backend::device,
    vk_sync,
};

pub trait ComputeImageLut {
    fn create(&mut self, device: &slingshot::Device) -> Image;
    fn compute(&mut self, rg: &mut rg::RenderGraph, img: &mut rg::Handle<Image>);
}

pub struct ImageLut {
    pub(crate) image: Arc<Image>,
    computer: Box<dyn ComputeImageLut>,
}

impl ImageLut {
    pub fn new(device: &slingshot::Device, mut computer: Box<dyn ComputeImageLut>) -> Self {
        Self {
            image: Arc::new(computer.create(device)),
            computer,
        }
    }

    pub fn compute(&mut self, rg: &mut rg::RenderGraph) {
        let mut rg_image = rg.import(self.image.clone(), vk_sync::AccessType::Nothing);

        self.computer.compute(rg, &mut rg_image);

        rg.export(
            rg_image,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
    }
}

//pub fn clear_depth(rg: &mut RenderGraph, img: &mut Handle<Image>) {
