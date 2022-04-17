#![allow(dead_code)]
use std::{mem::size_of, sync::Arc};

use glam::{IVec3, Vec3};
use kajiya_backend::{
    ash::vk,
    vk_sync::AccessType,
    vulkan::{
        buffer::{Buffer, BufferDesc},
        image::*,
        ray_tracing::RayTracingAcceleration,
        shader::{
            create_render_pass, PipelineShaderDesc, RasterPipelineDesc, RenderPass,
            RenderPassAttachmentDesc, RenderPassDesc, ShaderPipelineStage, ShaderSource,
        },
    },
    Device,
};
use kajiya_rg::{self as rg, GetOrCreateTemporal, SimpleRenderPass};
use rg::{BindMutToSimpleRenderPass, BindRgRef, IntoRenderPassPipelineBinding};
use rust_shaders_shared::frame_constants::{RcacheCascadeConstants, RCACHE_CASCADE_COUNT};
use vk::BufferUsageFlags;

use super::{wrc::WrcRenderState, GbufferDepth};

const MAX_SURFEL_CELLS: usize =
    RCACHE_CASCADE_SIZE * RCACHE_CASCADE_SIZE * RCACHE_CASCADE_SIZE * RCACHE_CASCADE_COUNT;
const MAX_SURFELS: usize = 1024 * 256;
const RCACHE_GRID_CELL_DIAMETER: f32 = 0.16;
const RCACHE_CASCADE_SIZE: usize = 32;

pub struct SurfelGiRenderState {
    surf_rcache_meta_buf: rg::Handle<Buffer>,

    surf_rcache_grid_meta_buf: rg::Handle<Buffer>,
    surf_rcache_grid_meta_buf2: rg::Handle<Buffer>,

    surf_rcache_entry_cell_buf: rg::Handle<Buffer>,
    surf_rcache_spatial_buf: rg::Handle<Buffer>,
    surf_rcache_irradiance_buf: rg::Handle<Buffer>,
    surf_rcache_aux_buf: rg::Handle<Buffer>,

    surf_rcache_life_buf: rg::Handle<Buffer>,
    surf_rcache_pool_buf: rg::Handle<Buffer>,

    surf_rcache_reposition_proposal_buf: rg::Handle<Buffer>,
    surf_rcache_reposition_proposal_count_buf: rg::Handle<Buffer>,

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
            .write(&mut self.surf_rcache_reposition_proposal_count_buf)
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

pub struct SurfelGiRenderer {
    debug_render_pass: Arc<RenderPass>,
    initialized: bool,
    cur_scroll: [IVec3; RCACHE_CASCADE_COUNT],
    prev_scroll: [IVec3; RCACHE_CASCADE_COUNT],
    parity: usize,
}

impl SurfelGiRenderer {
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        let debug_render_pass = create_render_pass(
            device,
            RenderPassDesc {
                color_attachments: &[
                    // view-space geometry normal; * 2 - 1 to decode
                    RenderPassAttachmentDesc::new(vk::Format::R32G32B32A32_SFLOAT),
                ],
                depth_attachment: Some(RenderPassAttachmentDesc::new(vk::Format::D32_SFLOAT)),
            },
        )?;

        Ok(Self {
            debug_render_pass,
            initialized: false,
            cur_scroll: Default::default(),
            prev_scroll: Default::default(),
            parity: 0,
        })
    }

    pub fn update_eye_position(&mut self, eye_position: Vec3) {
        for cascade in 0..RCACHE_CASCADE_COUNT {
            let cell_diameter = RCACHE_GRID_CELL_DIAMETER * (1 << cascade) as f32;
            let cascade_center = (eye_position / cell_diameter).floor().as_ivec3();
            let cascade_origin = cascade_center - IVec3::splat(RCACHE_CASCADE_SIZE as i32 / 2);

            self.prev_scroll[cascade] = self.cur_scroll[cascade];
            self.cur_scroll[cascade] = cascade_origin;
        }
    }

    pub fn constants(&self) -> [RcacheCascadeConstants; RCACHE_CASCADE_COUNT] {
        array_init::array_init(|cascade| {
            let cur_scroll = self.cur_scroll[cascade];
            let prev_scroll = self.prev_scroll[cascade];
            let scroll_amount = cur_scroll - prev_scroll;

            /*if scroll_amount.ne(&IVec3::ZERO) {
                log::info!("cascade {cascade} scrolled by {scroll_amount:?}");
            }*/

            RcacheCascadeConstants {
                origin: cur_scroll.extend(0),
                voxels_scrolled_this_frame: scroll_amount.extend(0),
            }
        })
    }
}

impl SurfelGiRenderer {
    pub fn allocate_surfels(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &mut GbufferDepth,
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
                size_of::<[u32; 2]>() * MAX_SURFEL_CELLS,
            ),
            surf_rcache_grid_meta_buf2: temporal_storage_buffer(
                rg,
                "surf_rcache.grid_meta_buf2",
                size_of::<[u32; 2]>() * MAX_SURFEL_CELLS,
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
                3 * size_of::<[f32; 4]>() * MAX_SURFELS,
            ),
            surf_rcache_aux_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.aux_buf",
                2 * 16 * size_of::<[f32; 4]>() * MAX_SURFELS,
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
            surf_rcache_reposition_proposal_count_buf: temporal_storage_buffer(
                rg,
                "surf_rcache.reposition_proposal_count_buf",
                size_of::<u32>() * MAX_SURFELS,
            ),
            debug_out: rg.create(gbuffer_desc.format(vk::Format::R32G32B32A32_SFLOAT)),
        };

        if 1 == self.parity {
            std::mem::swap(
                &mut state.surf_rcache_grid_meta_buf,
                &mut state.surf_rcache_grid_meta_buf2,
            );
        }

        if !self.initialized {
            SimpleRenderPass::new_compute(
                rg.add_pass("clear surfel pool"),
                "/shaders/surfel_gi/clear_surfel_pool.hlsl",
            )
            .write(&mut state.surf_rcache_pool_buf)
            .write(&mut state.surf_rcache_life_buf)
            .dispatch([MAX_SURFELS as _, 1, 1]);

            self.initialized = true;
        } else {
            SimpleRenderPass::new_compute(
                rg.add_pass("scroll cascades"),
                "/shaders/surfel_gi/scroll_cascades.hlsl",
            )
            .read(&mut state.surf_rcache_grid_meta_buf)
            .write(&mut state.surf_rcache_grid_meta_buf2)
            .write(&mut state.surf_rcache_entry_cell_buf)
            .write(&mut state.surf_rcache_irradiance_buf)
            .write(&mut state.surf_rcache_life_buf)
            .write(&mut state.surf_rcache_pool_buf)
            .write(&mut state.surf_rcache_meta_buf)
            .dispatch([
                RCACHE_CASCADE_SIZE as u32,
                RCACHE_CASCADE_SIZE as u32,
                (RCACHE_CASCADE_SIZE * RCACHE_CASCADE_COUNT) as u32,
            ]);

            std::mem::swap(
                &mut state.surf_rcache_grid_meta_buf,
                &mut state.surf_rcache_grid_meta_buf2,
            );

            self.parity = (self.parity + 1) % 2;
        }

        SimpleRenderPass::new_compute(
            rg.add_pass("find missing surfels"),
            "/shaders/surfel_gi/find_missing_surfels.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&gbuffer_depth.geometric_normal)
        .write(&mut state.surf_rcache_meta_buf)
        .write(&mut state.surf_rcache_grid_meta_buf)
        .write(&mut state.surf_rcache_entry_cell_buf)
        .read(&state.surf_rcache_spatial_buf)
        .read(&state.surf_rcache_irradiance_buf)
        .write(&mut state.debug_out)
        .write(&mut state.surf_rcache_pool_buf)
        .write(&mut state.surf_rcache_life_buf)
        .write(&mut state.surf_rcache_reposition_proposal_buf)
        .write(&mut state.surf_rcache_reposition_proposal_count_buf)
        .constants(gbuffer_desc.extent_inv_extent_2d())
        .dispatch(gbuffer_desc.extent);

        //state.draw_trace_origins(rg, self.debug_render_pass.clone(), gbuffer_depth);

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
        .write(&mut state.surf_rcache_reposition_proposal_count_buf)
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
        .write(&mut self.surf_rcache_reposition_proposal_buf)
        .write(&mut self.surf_rcache_reposition_proposal_count_buf)
        .bind(wrc)
        .write(&mut self.surf_rcache_meta_buf)
        .write(&mut self.surf_rcache_irradiance_buf)
        .write(&mut self.surf_rcache_aux_buf)
        .write(&mut self.surf_rcache_pool_buf)
        .write(&mut self.surf_rcache_entry_cell_buf)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays_indirect(tlas, &indirect_args_buf, 0);
    }

    fn draw_trace_origins(
        &mut self,
        rg: &mut rg::RenderGraph,
        render_pass: Arc<RenderPass>,
        gbuffer_depth: &mut GbufferDepth,
    ) {
        let mut pass = rg.add_pass("raster rcache origins");

        let pipeline = pass.register_raster_pipeline(
            &[
                PipelineShaderDesc::builder(ShaderPipelineStage::Vertex)
                    .hlsl_source("/shaders/surfel_gi/raster_origins_vs.hlsl")
                    .build()
                    .unwrap(),
                PipelineShaderDesc::builder(ShaderPipelineStage::Pixel)
                    .hlsl_source("/shaders/surfel_gi/raster_origins_ps.hlsl")
                    .build()
                    .unwrap(),
            ],
            RasterPipelineDesc::builder()
                .render_pass(render_pass.clone())
                .face_cull(true)
                .depth_write(false),
        );

        let depth_ref = pass.raster(
            &mut gbuffer_depth.depth,
            AccessType::DepthAttachmentWriteStencilReadOnly,
        );
        let color_ref = pass.raster(&mut self.debug_out, AccessType::ColorAttachmentWrite);

        let surf_rcache_meta_buf_ref = pass.read(
            &self.surf_rcache_meta_buf,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let surf_rcache_grid_meta_buf_ref = pass.read(
            &self.surf_rcache_grid_meta_buf,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let surf_rcache_life_buf_ref = pass.read(
            &self.surf_rcache_life_buf,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let surf_rcache_spatial_buf_ref = pass.read(
            &self.surf_rcache_spatial_buf,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        pass.render(move |api| {
            let [width, height, _] = color_ref.desc().extent;

            api.begin_render_pass(
                &*render_pass,
                [width, height],
                &[(color_ref, &ImageViewDesc::default())],
                Some((
                    depth_ref,
                    &ImageViewDesc::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .build()
                        .unwrap(),
                )),
            );

            api.set_default_view_and_scissor([width, height]);

            //let constants_offset = api.dynamic_constants().push(&(cascade_idx as u32));
            let _ = api.bind_raster_pipeline(pipeline.into_binding().descriptor_set(
                0,
                &[
                    surf_rcache_meta_buf_ref.bind(),
                    surf_rcache_grid_meta_buf_ref.bind(),
                    surf_rcache_life_buf_ref.bind(),
                    surf_rcache_spatial_buf_ref.bind(),
                    //rg::RenderPassBinding::DynamicConstants(constants_offset),
                ],
            ));

            unsafe {
                let raw_device = &api.device().raw;
                let cb = api.cb;

                raw_device.cmd_draw(
                    cb.raw,
                    // 6 verts (two triangles) per cube face
                    6 * 6 * MAX_SURFELS as u32,
                    1,
                    0,
                    0,
                );
            }

            api.end_render_pass();
        });
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
