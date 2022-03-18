#![allow(dead_code)]
use std::mem::size_of;

use kajiya_backend::{
    ash::vk,
    vulkan::{
        buffer::{Buffer, BufferDesc},
        image::*,
        ray_tracing::RayTracingAcceleration,
        shader::ShaderSource,
    },
};
use kajiya_rg::{self as rg, GetOrCreateTemporal, SimpleRenderPass};
use rg::BindMutToSimpleRenderPass;
use vk::BufferUsageFlags;

use super::{wrc::WrcRenderState, GbufferDepth};

const MAX_SURFEL_CELLS: usize = 1024 * 256;
const MAX_SURFELS: usize = MAX_SURFEL_CELLS;

pub struct SurfelGiRenderState {
    surf_rcache_meta_buf: rg::Handle<Buffer>,

    surf_rcache_grid_meta_buf: rg::Handle<Buffer>,

    surf_rcache_entry_cell_buf: rg::Handle<Buffer>,
    surf_rcache_spatial_buf: rg::Handle<Buffer>,
    surf_rcache_irradiance_buf: rg::Handle<Buffer>,
    surf_rcache_aux_buf: rg::Handle<Buffer>,

    surf_rcache_life_buf: rg::Handle<Buffer>,
    surf_rcache_pool_buf: rg::Handle<Buffer>,

    surf_rcache_reposition_proposal_buf: rg::Handle<Buffer>,

    pub debug_out: rg::Handle<Image>,
}

impl<'rg, RgPipelineHandle> BindMutToSimpleRenderPass<'rg, RgPipelineHandle>
    for SurfelGiRenderState
{
    fn bind_mut(
        &mut self,
        pass: SimpleRenderPass<'rg, RgPipelineHandle>,
    ) -> SimpleRenderPass<'rg, RgPipelineHandle> {
        pass.write(&mut self.surf_rcache_meta_buf)
            .write(&mut self.surf_rcache_pool_buf)
            .write(&mut self.surf_rcache_reposition_proposal_buf)
            .write(&mut self.surf_rcache_grid_meta_buf)
            .write(&mut self.surf_rcache_entry_cell_buf)
            .read(&self.surf_rcache_spatial_buf)
            .read(&self.surf_rcache_irradiance_buf)
            .write(&mut self.surf_rcache_life_buf)
    }
}

fn temporal_storage_buffer(
    rg: &mut rg::TemporalRenderGraph,
    name: &str,
    size: usize,
) -> rg::Handle<Buffer> {
    rg.get_or_create_temporal(
        name,
        BufferDesc::new(size, BufferUsageFlags::STORAGE_BUFFER),
    )
    .unwrap()
}

#[derive(Default)]
pub struct SurfelGiRenderer {
    initialized: bool,
}

impl SurfelGiRenderer {
    pub fn allocate_surfels(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        bent_normals: &rg::Handle<Image>,
        gbuffer_depth: &GbufferDepth,
    ) -> SurfelGiRenderState {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let mut state = SurfelGiRenderState {
            // 0: hash grid cell count
            // 1: surfel count
            surf_rcache_meta_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.meta_buf",
                size_of::<u32>() * 8,
            ),
            surf_rcache_grid_meta_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.grid_meta_buf",
                size_of::<[u32; 4]>() * MAX_SURFEL_CELLS,
            ),
            surf_rcache_entry_cell_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.entry_cell_buf",
                size_of::<u32>() * MAX_SURFELS,
            ),
            surf_rcache_spatial_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.spatial_buf",
                size_of::<[f32; 4]>() * MAX_SURFELS,
            ),
            surf_rcache_irradiance_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.irradiance_buf",
                size_of::<[f32; 4]>() * MAX_SURFELS,
            ),
            surf_rcache_aux_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.aux_buf",
                3 * size_of::<[f32; 4]>() * MAX_SURFELS,
            ),
            surf_rcache_life_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.life_buf",
                size_of::<u32>() * MAX_SURFELS,
            ),
            surf_rcache_pool_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.pool_buf",
                size_of::<u32>() * MAX_SURFELS,
            ),
            surf_rcache_reposition_proposal_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.reposition_proposal_buf",
                size_of::<[f32; 4]>() * MAX_SURFELS,
            ),
            debug_out: rg.create(gbuffer_desc.format(vk::Format::R32G32B32A32_SFLOAT)),
        };

        if !self.initialized {
            SimpleRenderPass::new_compute(
                rg.add_pass("clear surfel pool"),
                "/shaders/surfel_gi/clear_surfel_pool.hlsl",
            )
            .write(&mut state.surf_rcache_pool_buf)
            .write(&mut state.surf_rcache_life_buf)
            .dispatch([MAX_SURFELS as _, 1, 1]);

            self.initialized = true;
        }

        SimpleRenderPass::new_compute(
            rg.add_pass("find missing surfels"),
            "/shaders/surfel_gi/find_missing_surfels.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(bent_normals)
        .write(&mut state.surf_rcache_meta_buf)
        .write(&mut state.surf_rcache_grid_meta_buf)
        .write(&mut state.surf_rcache_entry_cell_buf)
        .read(&state.surf_rcache_spatial_buf)
        .read(&state.surf_rcache_irradiance_buf)
        .write(&mut state.debug_out)
        .write(&mut state.surf_rcache_pool_buf)
        .write(&mut state.surf_rcache_life_buf)
        .write(&mut state.surf_rcache_reposition_proposal_buf)
        .constants(gbuffer_desc.extent_inv_extent_2d())
        .dispatch(gbuffer_desc.extent);

        let indirect_args_buf = {
            let mut indirect_args_buf = rg.create(BufferDesc::new(
                (size_of::<u32>() * 4) * 2,
                vk::BufferUsageFlags::empty(),
            ));

            SimpleRenderPass::new_compute(
                rg.add_pass("surfel dispatch args"),
                "/shaders/surfel_gi/prepare_surfel_assignment_dispatch_args.hlsl",
            )
            .read(&state.surf_rcache_meta_buf)
            .write(&mut indirect_args_buf)
            .dispatch([1, 1, 1]);

            indirect_args_buf
        };

        SimpleRenderPass::new_compute(
            rg.add_pass("age surfels"),
            "/shaders/surfel_gi/age_surfels.hlsl",
        )
        .write(&mut state.surf_rcache_meta_buf)
        .write(&mut state.surf_rcache_grid_meta_buf)
        .write(&mut state.surf_rcache_entry_cell_buf)
        .write(&mut state.surf_rcache_life_buf)
        .write(&mut state.surf_rcache_pool_buf)
        .write(&mut state.surf_rcache_spatial_buf)
        .write(&mut state.surf_rcache_reposition_proposal_buf)
        .write(&mut state.surf_rcache_irradiance_buf)
        .dispatch_indirect(&indirect_args_buf, 0);

        state
    }
}

impl SurfelGiRenderState {
    pub fn trace_irradiance(
        &mut self,
        rg: &mut rg::RenderGraph,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
        wrc: &WrcRenderState,
    ) {
        let indirect_args_buf = {
            let mut indirect_args_buf = rg.create(BufferDesc::new(
                size_of::<u32>() * 4,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ));

            SimpleRenderPass::new_compute(
                rg.add_pass("surfel gi dispatch args"),
                "/shaders/surfel_gi/prepare_trace_dispatch_args.hlsl",
            )
            .read(&self.surf_rcache_meta_buf)
            .write(&mut indirect_args_buf)
            .dispatch([1, 1, 1]);

            indirect_args_buf
        };

        SimpleRenderPass::new_rt(
            rg.add_pass("surfel gi trace"),
            ShaderSource::hlsl("/shaders/surfel_gi/trace_irradiance.rgen.hlsl"),
            [
                ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
            ],
            [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
        )
        .read(&self.surf_rcache_spatial_buf)
        .read(sky_cube)
        .write(&mut self.surf_rcache_grid_meta_buf)
        .read(&self.surf_rcache_life_buf)
        .read(&self.surf_rcache_reposition_proposal_buf)
        .bind(wrc)
        .write(&mut self.surf_rcache_meta_buf)
        .write(&mut self.surf_rcache_irradiance_buf)
        .write(&mut self.surf_rcache_aux_buf)
        .write(&mut self.surf_rcache_pool_buf)
        .write(&mut self.surf_rcache_entry_cell_buf)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays_indirect(tlas, &indirect_args_buf, 0);
    }
}

/*fn inclusive_prefix_scan_u32_1m(rg: &mut rg::RenderGraph, input_buf: &mut rg::Handle<Buffer>) {
    const SEGMENT_SIZE: usize = 1024;

    SimpleRenderPass::new_compute(
        rg.add_pass("prefix scan 1"),
        "/shaders/surfel_gi/inclusive_prefix_scan.hlsl",
    )
    .write(input_buf)
    .dispatch([(SEGMENT_SIZE * SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect

    let mut segment_sum_buf = rg.create(BufferDesc::new(
        size_of::<u32>() * SEGMENT_SIZE,
        vk::BufferUsageFlags::empty(),
    ));
    SimpleRenderPass::new_compute(
        rg.add_pass("prefix scan 2"),
        "/shaders/surfel_gi/inclusive_prefix_scan_segments.hlsl",
    )
    .read(input_buf)
    .write(&mut segment_sum_buf)
    .dispatch([(SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect

    SimpleRenderPass::new_compute(
        rg.add_pass("prefix scan merge"),
        "/shaders/surfel_gi/inclusive_prefix_scan_merge.hlsl",
    )
    .write(input_buf)
    .read(&segment_sum_buf)
    .dispatch([(SEGMENT_SIZE * SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect
}*/
