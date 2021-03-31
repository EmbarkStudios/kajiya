#![allow(dead_code)]
use std::mem::size_of;

use kajiya_backend::{
    ash::vk,
    vulkan::{
        buffer::{Buffer, BufferDesc},
        image::*,
        ray_tracing::RayTracingAcceleration,
    },
};
use kajiya_rg::{self as rg, GetOrCreateTemporal, SimpleRenderPass};
use vk::BufferUsageFlags;

use super::GbufferDepth;

const MAX_SURFEL_CELLS: usize = 1024 * 1024;
const MAX_SURFELS: usize = MAX_SURFEL_CELLS;
const MAX_SURFELS_PER_CELL: usize = 32;

pub struct SurfelGiRenderState {
    surfel_meta_buf: rg::Handle<Buffer>,
    surfel_hash_key_buf: rg::Handle<Buffer>,
    surfel_hash_value_buf: rg::Handle<Buffer>,

    cell_index_offset_buf: rg::Handle<Buffer>,
    surfel_index_buf: rg::Handle<Buffer>,

    surfel_spatial_buf: rg::Handle<Buffer>,
    surfel_irradiance_buf: rg::Handle<Buffer>,
    surfel_sh_buf: rg::Handle<Buffer>,

    pub debug_out: rg::Handle<Image>,
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

pub fn allocate_surfels(
    rg: &mut rg::TemporalRenderGraph,
    bent_normals: &rg::Handle<Image>,
    gbuffer_depth: &GbufferDepth,
) -> SurfelGiRenderState {
    let gbuffer_desc = gbuffer_depth.gbuffer.desc();

    let mut state = SurfelGiRenderState {
        // 0: hash grid cell count
        // 1: surfel count
        surfel_meta_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.surfel_meta_buf",
            size_of::<u32>() * 8,
        ),
        surfel_hash_key_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.surfel_hash_key_buf",
            size_of::<u32>() * MAX_SURFEL_CELLS,
        ),
        surfel_hash_value_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.surfel_hash_value_buf",
            size_of::<u32>() * MAX_SURFEL_CELLS,
        ),
        cell_index_offset_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.cell_index_offset_buf",
            size_of::<u32>() * (MAX_SURFEL_CELLS + 1),
        ),
        surfel_index_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.surfel_index_buf",
            size_of::<u32>() * MAX_SURFEL_CELLS * MAX_SURFELS_PER_CELL,
        ),
        surfel_spatial_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.surfel_spatial_buf",
            size_of::<[f32; 4]>() * MAX_SURFELS,
        ),
        surfel_irradiance_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.surfel_irradiance_buf",
            size_of::<[f32; 4]>() * MAX_SURFELS,
        ),
        surfel_sh_buf: temporal_storage_buffer(
            rg,
            "surfel_gi.surfel_sh_buf",
            3 * size_of::<[f32; 4]>() * MAX_SURFELS,
        ),
        //surfel_sh_buf: temporal_storage_buffer(rg, "surfel_gi.surfel_sh_buf", 4 * 32 * MAX_SURFELS),
        debug_out: rg.create(gbuffer_desc.format(vk::Format::R32G32B32A32_SFLOAT)),
    };

    let mut tile_surfel_alloc_tex = rg.create(
        gbuffer_desc
            .div_up_extent([8, 8, 1])
            .format(vk::Format::R32G32_UINT),
    );

    SimpleRenderPass::new_compute(
        rg.add_pass("find missing surfels"),
        "/assets/shaders/surfel_gi/find_missing_surfels.hlsl",
    )
    .read(&gbuffer_depth.gbuffer)
    .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
    .read(bent_normals)
    .write(&mut state.surfel_meta_buf)
    .write(&mut state.surfel_hash_key_buf)
    .write(&mut state.surfel_hash_value_buf)
    .read(&state.cell_index_offset_buf)
    .read(&state.surfel_index_buf)
    .read(&state.surfel_spatial_buf)
    .read(&state.surfel_irradiance_buf)
    .read(&state.surfel_sh_buf)
    .write(&mut tile_surfel_alloc_tex)
    .write(&mut state.debug_out)
    .constants(gbuffer_desc.extent_inv_extent_2d())
    .dispatch(gbuffer_desc.extent);

    SimpleRenderPass::new_compute(
        rg.add_pass("allocate surfels"),
        "/assets/shaders/surfel_gi/allocate_surfels.hlsl",
    )
    .read(&gbuffer_depth.gbuffer)
    .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
    .read(&state.surfel_meta_buf)
    .read(&state.surfel_hash_key_buf)
    .read(&state.surfel_hash_value_buf)
    .read(&state.surfel_index_buf)
    .write(&mut state.surfel_spatial_buf)
    .read(&tile_surfel_alloc_tex)
    .constants(gbuffer_desc.extent_inv_extent_2d())
    .dispatch(tile_surfel_alloc_tex.desc().extent);

    state.assign_surfels_to_grid_cells(rg);

    state
}

impl SurfelGiRenderState {
    fn assign_surfels_to_grid_cells(&mut self, rg: &mut rg::RenderGraph) {
        let indirect_args_buf = {
            let mut indirect_args_buf = rg.create(BufferDesc::new(
                (size_of::<u32>() * 4) * 2,
                vk::BufferUsageFlags::empty(),
            ));

            SimpleRenderPass::new_compute(
                rg.add_pass("surfel dispatch args"),
                "/assets/shaders/surfel_gi/prepare_surfel_assignment_dispatch_args.hlsl",
            )
            .read(&self.surfel_meta_buf)
            .write(&mut indirect_args_buf)
            .dispatch([1, 1, 1]);

            indirect_args_buf
        };

        SimpleRenderPass::new_compute(
            rg.add_pass("clear surfel cells"),
            "/assets/shaders/surfel_gi/clear_cells.hlsl",
        )
        .write(&mut self.cell_index_offset_buf)
        .dispatch_indirect(&indirect_args_buf, 0);

        SimpleRenderPass::new_compute(
            rg.add_pass("count surfels per cell"),
            "/assets/shaders/surfel_gi/count_surfels_per_cell.hlsl",
        )
        .read(&self.surfel_meta_buf)
        .read(&self.surfel_hash_key_buf)
        .read(&self.surfel_hash_value_buf)
        .read(&self.surfel_spatial_buf)
        .write(&mut self.cell_index_offset_buf)
        // Thread per surfel
        .dispatch_indirect(&indirect_args_buf, 16);

        inclusive_prefix_scan_u32_1m(rg, &mut self.cell_index_offset_buf);
        // TODO: prefix-scan

        SimpleRenderPass::new_compute(
            rg.add_pass("slot surfels into cells"),
            "/assets/shaders/surfel_gi/slot_surfels_into_cells.hlsl",
        )
        .read(&self.surfel_meta_buf)
        .read(&self.surfel_hash_key_buf)
        .read(&self.surfel_hash_value_buf)
        .read(&self.surfel_spatial_buf)
        .write(&mut self.cell_index_offset_buf)
        .write(&mut self.surfel_index_buf)
        // Thread per surfel
        .dispatch_indirect(&indirect_args_buf, 16);
    }

    pub fn trace_irradiance(
        &mut self,
        rg: &mut rg::RenderGraph,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) {
        let indirect_args_buf = {
            let mut indirect_args_buf = rg.create(BufferDesc::new(
                size_of::<u32>() * 4,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            ));

            SimpleRenderPass::new_compute(
                rg.add_pass("surfel gi dispatch args"),
                "/assets/shaders/surfel_gi/prepare_trace_dispatch_args.hlsl",
            )
            .read(&self.surfel_meta_buf)
            .write(&mut indirect_args_buf)
            .dispatch([1, 1, 1]);

            indirect_args_buf
        };

        SimpleRenderPass::new_rt(
            rg.add_pass("surfel gi trace"),
            "/assets/shaders/surfel_gi/trace_irradiance.rgen.hlsl",
            &[
                "/assets/shaders/rt/gbuffer.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/gbuffer.rchit.hlsl"],
        )
        .read(&self.surfel_spatial_buf)
        .write(&mut self.surfel_irradiance_buf)
        .write(&mut self.surfel_sh_buf)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays_indirect(tlas, &indirect_args_buf, 0);
    }
}

fn inclusive_prefix_scan_u32_1m(rg: &mut rg::RenderGraph, input_buf: &mut rg::Handle<Buffer>) {
    const SEGMENT_SIZE: usize = 1024;

    SimpleRenderPass::new_compute(
        rg.add_pass("prefix scan 1"),
        "/assets/shaders/surfel_gi/inclusive_prefix_scan.hlsl",
    )
    .write(input_buf)
    .dispatch([(SEGMENT_SIZE * SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect

    let mut segment_sum_buf = rg.create(BufferDesc::new(
        size_of::<u32>() * SEGMENT_SIZE,
        vk::BufferUsageFlags::empty(),
    ));
    SimpleRenderPass::new_compute(
        rg.add_pass("prefix scan 2"),
        "/assets/shaders/surfel_gi/inclusive_prefix_scan_segments.hlsl",
    )
    .read(input_buf)
    .write(&mut segment_sum_buf)
    .dispatch([(SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect

    SimpleRenderPass::new_compute(
        rg.add_pass("prefix scan merge"),
        "/assets/shaders/surfel_gi/inclusive_prefix_scan_merge.hlsl",
    )
    .write(input_buf)
    .read(&segment_sum_buf)
    .dispatch([(SEGMENT_SIZE * SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect
}
