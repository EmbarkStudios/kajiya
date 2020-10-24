use crate::{
    backend::{self, image::*, shader::*, RenderBackend},
    render_passes::SdfRasterBricks,
    renderer::*,
    rg,
    rg::RetiredRenderGraph,
};
use ash::vk;
use backend::buffer::{Buffer, BufferDesc};
use byte_slice_cast::AsByteSlice;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;

pub struct SdfRenderClient {
    raster_simple_render_pass: Arc<RenderPass>,
    sdf_img: TemporalImage,
    cube_index_buffer: Arc<Buffer>,
    frame_idx: u32,
}

impl SdfRenderClient {
    pub fn new(backend: &RenderBackend) -> anyhow::Result<Self> {
        let sdf_img = backend.device.create_image(
            ImageDesc::new_3d(vk::Format::R16_SFLOAT, [SDF_DIM, SDF_DIM, SDF_DIM])
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED),
            None,
        )?;

        let cube_indices = cube_indices();
        let cube_index_buffer = backend.device.create_buffer(
            BufferDesc {
                size: cube_indices.len() * 4,
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
            },
            Some((&cube_indices).as_byte_slice()),
        )?;

        let raster_simple_render_pass = create_render_pass(
            &*backend.device,
            RenderPassDesc {
                color_attachments: &[RenderPassAttachmentDesc::new(
                    vk::Format::R16G16B16A16_SFLOAT,
                )
                .garbage_input()],
                depth_attachment: Some(RenderPassAttachmentDesc::new(
                    vk::Format::D24_UNORM_S8_UINT,
                )),
            },
        )?;

        Ok(Self {
            raster_simple_render_pass,

            sdf_img: TemporalImage::new(Arc::new(sdf_img)),
            cube_index_buffer: Arc::new(cube_index_buffer),
            frame_idx: 0u32,
        })
    }
}

impl RenderClient for SdfRenderClient {
    fn prepare_render_graph(
        &mut self,
        rg: &mut crate::rg::RenderGraph,
        frame_state: &crate::FrameState,
    ) -> rg::ExportedHandle<Image> {
        let mut sdf_img = rg.import_image(self.sdf_img.resource.clone(), self.sdf_img.access_type);
        let cube_index_buffer = rg.import_buffer(
            self.cube_index_buffer.clone(),
            vk_sync::AccessType::TransferWrite,
        );

        let mut depth_img = crate::render_passes::create_image(
            rg,
            ImageDesc::new_2d(vk::Format::D24_UNORM_S8_UINT, frame_state.window_cfg.dims()),
        );
        crate::render_passes::clear_depth(rg, &mut depth_img);
        crate::render_passes::edit_sdf(rg, &mut sdf_img, self.frame_idx == 0);

        let sdf_raster_bricks: SdfRasterBricks =
            crate::render_passes::calculate_sdf_bricks_meta(rg, &sdf_img);

        /*let mut tex = crate::render_passes::raymarch_sdf(
            rg,
            &sdf_img,
            ImageDesc::new_2d(
                vk::Format::R16G16B16A16_SFLOAT,
                frame_state.window_cfg.dims(),
            ),
        );*/
        let mut tex = crate::render_passes::create_image(
            rg,
            ImageDesc::new_2d(
                vk::Format::R16G16B16A16_SFLOAT,
                frame_state.window_cfg.dims(),
            ),
        );
        crate::render_passes::clear_color(rg, &mut tex, [0.1, 0.2, 0.5, 1.0]);

        crate::render_passes::raster_sdf(
            rg,
            self.raster_simple_render_pass.clone(),
            &mut depth_img,
            &mut tex,
            crate::render_passes::RasterSdfData {
                sdf_img: &sdf_img,
                brick_inst_buffer: &sdf_raster_bricks.brick_inst_buffer,
                brick_meta_buffer: &sdf_raster_bricks.brick_meta_buffer,
                cube_index_buffer: &cube_index_buffer,
            },
        );

        let tex = crate::render_passes::blur(rg, &tex);
        self.sdf_img.last_rg_handle = Some(rg.export_image(sdf_img, vk::ImageUsageFlags::empty()));

        rg.export_image(tex, vk::ImageUsageFlags::SAMPLED)
    }

    fn retire_render_graph(&mut self, retired_rg: &RetiredRenderGraph) {
        if let Some(handle) = self.sdf_img.last_rg_handle.take() {
            self.sdf_img.access_type = retired_rg.get_image(handle).1;
        }

        self.frame_idx = self.frame_idx.overflowing_add(1).0;
    }
}

// Vertices: bits 0, 1, 2, map to +/- X, Y, Z
fn cube_indices() -> Vec<u32> {
    let mut res = Vec::with_capacity(6 * 2 * 3);

    for (ndim, dim0, dim1) in [(1, 2, 4), (2, 4, 1), (4, 1, 2)].iter().copied() {
        for (nbit, dim0, dim1) in [(0, dim1, dim0), (ndim, dim0, dim1)].iter().copied() {
            res.push(nbit);
            res.push(nbit + dim0);
            res.push(nbit + dim1);

            res.push(nbit + dim1);
            res.push(nbit + dim0);
            res.push(nbit + dim0 + dim1);
        }
    }

    res
}

struct TemporalImage {
    resource: Arc<Image>,
    access_type: vk_sync::AccessType,
    last_rg_handle: Option<rg::ExportedHandle<Image>>,
}

impl TemporalImage {
    pub fn new(resource: Arc<Image>) -> Self {
        Self {
            resource,
            access_type: vk_sync::AccessType::Nothing,
            last_rg_handle: None,
        }
    }
}
