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

struct ImageLut {
    image: Arc<Image>,
}

impl ImageLut {
    pub fn compute(&mut self, rg: &mut crate::rg::RenderGraph) {
        let mut rg_image = rg.import_image(
            self.image.clone(),
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        rg.export_image(
            rg_image,
            vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
    }
}
