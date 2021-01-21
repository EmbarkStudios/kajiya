use std::sync::Arc;

use slingshot::{
    ash::vk,
    backend::{
        buffer::{Buffer, BufferDesc},
        device,
        image::*,
        shader::*,
        RenderBackend,
    },
    rg,
    vk_sync::AccessType,
};

use crate::temporal::*;

pub struct SurfelGiRenderer {
    surfel_meta: Temporal<Buffer>,
    surfel_hash: Temporal<Buffer>,
    surfel_spatial: Temporal<Buffer>,
}

const MAX_SURFEL_CELLS: usize = 1024 * 1024;
const MAX_SURFELS: usize = MAX_SURFEL_CELLS;

impl SurfelGiRenderer {
    pub fn new(device: &device::Device) -> Self {
        let surfel_meta = device
            .create_buffer(
                BufferDesc::new(4, vk::BufferUsageFlags::STORAGE_BUFFER),
                None,
            )
            .unwrap();

        let surfel_hash = device
            .create_buffer(
                BufferDesc::new(
                    4 * 2 * MAX_SURFEL_CELLS,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                None,
            )
            .unwrap();

        let surfel_spatial = device
            .create_buffer(
                BufferDesc::new(16 * MAX_SURFELS, vk::BufferUsageFlags::STORAGE_BUFFER),
                None,
            )
            .unwrap();

        Self {
            surfel_meta: Temporal::new(Arc::new(surfel_meta)),
            surfel_hash: Temporal::new(Arc::new(surfel_hash)),
            surfel_spatial: Temporal::new(Arc::new(surfel_spatial)),
        }
    }

    pub fn begin(&mut self, rg: &mut rg::RenderGraph) -> SurfelGiRenderInstance {
        SurfelGiRenderInstance {
            surfel_meta: rg.import_temporal(&mut self.surfel_meta),
            surfel_hash: rg.import_temporal(&mut self.surfel_hash),
            surfel_spatial: rg.import_temporal(&mut self.surfel_spatial),
        }
    }

    pub fn end(&mut self, rg: &mut rg::RenderGraph, inst: SurfelGiRenderInstance) {
        rg.export_temporal(inst.surfel_meta, &mut self.surfel_meta);
        rg.export_temporal(inst.surfel_hash, &mut self.surfel_hash);
        rg.export_temporal(inst.surfel_spatial, &mut self.surfel_spatial);
    }

    pub fn retire(&mut self, rg: &rg::RetiredRenderGraph) {
        rg.retire_temporal(&mut self.surfel_meta);
        rg.retire_temporal(&mut self.surfel_hash);
        rg.retire_temporal(&mut self.surfel_spatial);
    }
}

pub struct SurfelGiRenderInstance {
    pub surfel_meta: rg::Handle<Buffer>,
    pub surfel_hash: rg::Handle<Buffer>,
    pub surfel_spatial: rg::Handle<Buffer>,
}

impl SurfelGiRenderInstance {
    pub fn allocate_surfels(
        &mut self,
        rg: &mut rg::RenderGraph,
        gbuffer: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let mut pass = rg.add_pass();

        let pipeline =
            pass.register_compute_pipeline("/assets/shaders/surfel_gi/allocate_surfels.hlsl");

        let gbuffer_ref = pass.read(
            gbuffer,
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        );
        let depth_ref = pass.read(
            depth,
            AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        );

        let mut debug_out = pass.create(&gbuffer.desc().format(vk::Format::R32G32B32A32_SFLOAT));
        let debug_out_ref = pass.write(&mut debug_out, AccessType::ComputeShaderWrite);

        let surfel_meta_ref = pass.write(&mut self.surfel_meta, AccessType::ComputeShaderWrite);
        let surfel_hash_ref = pass.write(&mut self.surfel_hash, AccessType::ComputeShaderWrite);
        let surfel_spatial_ref =
            pass.write(&mut self.surfel_spatial, AccessType::ComputeShaderWrite);

        pass.render(move |api| {
            let pipeline = api.bind_compute_pipeline(pipeline.into_binding().descriptor_set(
                0,
                &[
                    gbuffer_ref.bind(),
                    depth_ref.bind_view(
                        ImageViewDescBuilder::default().aspect_mask(vk::ImageAspectFlags::DEPTH),
                    ),
                    surfel_meta_ref.bind(),
                    surfel_hash_ref.bind(),
                    surfel_spatial_ref.bind(),
                    debug_out_ref.bind(),
                ],
            ));

            pipeline.dispatch(gbuffer_ref.desc().extent);
        });

        debug_out
    }
}
