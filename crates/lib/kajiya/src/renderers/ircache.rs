#![allow(dead_code)]
use std::{mem::size_of, sync::Arc};

use glam::{IVec3, Vec3};
use kajiya_backend::{
    ash::vk,
    vulkan::{
        buffer::{Buffer, BufferDesc},
        image::*,
        ray_tracing::RayTracingAcceleration,
        shader::{
            create_render_pass, RenderPass, RenderPassAttachmentDesc, RenderPassDesc, ShaderSource,
        },
    },
    Device,
};
use kajiya_rg::{self as rg, GetOrCreateTemporal, SimpleRenderPass};
use rg::BindMutToSimpleRenderPass;
use rust_shaders_shared::frame_constants::{IrcacheCascadeConstants, IRCACHE_CASCADE_COUNT};
use vk::BufferUsageFlags;

use crate::renderers::prefix_scan::inclusive_prefix_scan_u32_1m;

use super::wrc::WrcRenderState;

const MAX_GRID_CELLS: usize =
    IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_COUNT;
const MAX_ENTRIES: usize = 1024 * 64;

// Must match GPU side
const IRCACHE_GRID_CELL_DIAMETER: f32 = 0.16 * 0.125;
const IRCACHE_CASCADE_SIZE: usize = 32;
const IRCACHE_SAMPLES_PER_FRAME: usize = 4;
const IRCACHE_VALIDATION_SAMPLES_PER_FRAME: usize = 4;

pub struct IrcacheRenderState {
    ircache_meta_buf: rg::Handle<Buffer>,

    ircache_grid_meta_buf: rg::Handle<Buffer>,
    ircache_grid_meta_buf2: rg::Handle<Buffer>,

    ircache_entry_cell_buf: rg::Handle<Buffer>,
    ircache_spatial_buf: rg::Handle<Buffer>,
    ircache_irradiance_buf: rg::Handle<Buffer>,
    ircache_aux_buf: rg::Handle<Buffer>,

    ircache_life_buf: rg::Handle<Buffer>,
    ircache_pool_buf: rg::Handle<Buffer>,
    ircache_entry_indirection_buf: rg::Handle<Buffer>,

    ircache_reposition_proposal_buf: rg::Handle<Buffer>,
    ircache_reposition_proposal_count_buf: rg::Handle<Buffer>,

    pending_irradiance_sum: bool,
}

impl<'rg, RgPipelineHandle> BindMutToSimpleRenderPass<'rg, RgPipelineHandle>
    for IrcacheRenderState
{
    fn bind_mut(
        &mut self,
        pass: SimpleRenderPass<'rg, RgPipelineHandle>,
    ) -> SimpleRenderPass<'rg, RgPipelineHandle> {
        assert!(!self.pending_irradiance_sum);

        pass.write_no_sync(&mut self.ircache_meta_buf)
            .write_no_sync(&mut self.ircache_pool_buf)
            .write_no_sync(&mut self.ircache_reposition_proposal_buf)
            .write_no_sync(&mut self.ircache_reposition_proposal_count_buf)
            .write_no_sync(&mut self.ircache_grid_meta_buf)
            .write_no_sync(&mut self.ircache_entry_cell_buf)
            .read(&self.ircache_spatial_buf)
            .read(&self.ircache_irradiance_buf)
            .write_no_sync(&mut self.ircache_life_buf)
    }
}

fn temporal_storage_buffer(
    rg: &mut rg::TemporalRenderGraph,
    name: &str,
    size: usize,
) -> rg::Handle<Buffer> {
    rg.get_or_create_temporal(
        name,
        BufferDesc::new_gpu_only(size, BufferUsageFlags::STORAGE_BUFFER),
    )
    .unwrap()
}

pub struct IrcacheRenderer {
    debug_render_pass: Arc<RenderPass>,
    initialized: bool,
    grid_center: Vec3,
    cur_scroll: [IVec3; IRCACHE_CASCADE_COUNT],
    prev_scroll: [IVec3; IRCACHE_CASCADE_COUNT],
    parity: usize,
    pub enable_scroll: bool,
}

impl IrcacheRenderer {
    pub fn new(device: &Device) -> Self {
        let debug_render_pass = create_render_pass(
            device,
            RenderPassDesc {
                color_attachments: &[
                    // view-space geometry normal; * 2 - 1 to decode
                    RenderPassAttachmentDesc::new(vk::Format::R32G32B32A32_SFLOAT),
                ],
                depth_attachment: Some(RenderPassAttachmentDesc::new(vk::Format::D32_SFLOAT)),
            },
        );

        Self {
            debug_render_pass,
            initialized: false,
            grid_center: Vec3::ZERO,
            cur_scroll: Default::default(),
            prev_scroll: Default::default(),
            parity: 0,
            enable_scroll: true,
        }
    }

    pub fn update_eye_position(&mut self, eye_position: Vec3) {
        if !self.enable_scroll {
            return;
        }

        self.grid_center = eye_position;

        for cascade in 0..IRCACHE_CASCADE_COUNT {
            let cell_diameter = IRCACHE_GRID_CELL_DIAMETER * (1 << cascade) as f32;
            let cascade_center = (eye_position / cell_diameter).floor().as_ivec3();
            let cascade_origin = cascade_center - IVec3::splat(IRCACHE_CASCADE_SIZE as i32 / 2);

            self.prev_scroll[cascade] = self.cur_scroll[cascade];
            self.cur_scroll[cascade] = cascade_origin;
        }
    }

    pub fn constants(&self) -> [IrcacheCascadeConstants; IRCACHE_CASCADE_COUNT] {
        array_init::array_init(|cascade| {
            let cur_scroll = self.cur_scroll[cascade];
            let prev_scroll = self.prev_scroll[cascade];
            let scroll_amount = cur_scroll - prev_scroll;

            /*if scroll_amount.ne(&IVec3::ZERO) {
                log::info!("cascade {cascade} scrolled by {scroll_amount:?}");
            }*/

            IrcacheCascadeConstants {
                origin: cur_scroll.extend(0),
                voxels_scrolled_this_frame: scroll_amount.extend(0),
            }
        })
    }

    pub fn grid_center(&self) -> Vec3 {
        self.grid_center
    }
}

impl IrcacheRenderer {
    #[allow(clippy::assertions_on_constants)]
    #[allow(clippy::manual_bits)] // multiplying by 8 is not always for bits, Clippy
    pub fn prepare(&mut self, rg: &mut rg::TemporalRenderGraph) -> IrcacheRenderState {
        const INDIRECTION_BUF_ELEM_COUNT: usize = 1024 * 1024;
        assert!(INDIRECTION_BUF_ELEM_COUNT >= MAX_ENTRIES);

        let mut state = IrcacheRenderState {
            // 0: hash grid cell count
            // 1: entry count
            ircache_meta_buf: temporal_storage_buffer(rg, "ircache.meta_buf", size_of::<u32>() * 8),
            ircache_grid_meta_buf: temporal_storage_buffer(
                rg,
                "ircache.grid_meta_buf",
                size_of::<[u32; 2]>() * MAX_GRID_CELLS,
            ),
            ircache_grid_meta_buf2: temporal_storage_buffer(
                rg,
                "ircache.grid_meta_buf2",
                size_of::<[u32; 2]>() * MAX_GRID_CELLS,
            ),
            ircache_entry_cell_buf: temporal_storage_buffer(
                rg,
                "ircache.entry_cell_buf",
                size_of::<u32>() * MAX_ENTRIES,
            ),
            ircache_spatial_buf: temporal_storage_buffer(
                rg,
                "ircache.spatial_buf",
                size_of::<[f32; 4]>() * MAX_ENTRIES,
            ),
            ircache_irradiance_buf: temporal_storage_buffer(
                rg,
                "ircache.irradiance_buf",
                3 * size_of::<[f32; 4]>() * MAX_ENTRIES,
            ),
            ircache_aux_buf: temporal_storage_buffer(
                rg,
                "ircache.aux_buf",
                4 * 16 * size_of::<[f32; 4]>() * MAX_ENTRIES,
            ),
            ircache_life_buf: temporal_storage_buffer(
                rg,
                "ircache.life_buf",
                size_of::<u32>() * MAX_ENTRIES,
            ),
            ircache_pool_buf: temporal_storage_buffer(
                rg,
                "ircache.pool_buf",
                size_of::<u32>() * MAX_ENTRIES,
            ),
            ircache_entry_indirection_buf: temporal_storage_buffer(
                rg,
                "ircache.entry_indirection_buf",
                size_of::<u32>() * INDIRECTION_BUF_ELEM_COUNT,
            ),
            ircache_reposition_proposal_buf: temporal_storage_buffer(
                rg,
                "ircache.reposition_proposal_buf",
                size_of::<[f32; 4]>() * MAX_ENTRIES,
            ),
            ircache_reposition_proposal_count_buf: temporal_storage_buffer(
                rg,
                "ircache.reposition_proposal_count_buf",
                size_of::<u32>() * MAX_ENTRIES,
            ),
            pending_irradiance_sum: false,
        };

        if 1 == self.parity {
            std::mem::swap(
                &mut state.ircache_grid_meta_buf,
                &mut state.ircache_grid_meta_buf2,
            );
        }

        if !self.initialized {
            SimpleRenderPass::new_compute(
                rg.add_pass("clear ircache pool"),
                "/shaders/ircache/clear_ircache_pool.hlsl",
            )
            .write(&mut state.ircache_pool_buf)
            .write(&mut state.ircache_life_buf)
            .dispatch([MAX_ENTRIES as _, 1, 1]);

            self.initialized = true;
        } else {
            SimpleRenderPass::new_compute(
                rg.add_pass("scroll cascades"),
                "/shaders/ircache/scroll_cascades.hlsl",
            )
            .read(&state.ircache_grid_meta_buf)
            .write(&mut state.ircache_grid_meta_buf2)
            .write(&mut state.ircache_entry_cell_buf)
            .write(&mut state.ircache_irradiance_buf)
            .write(&mut state.ircache_life_buf)
            .write(&mut state.ircache_pool_buf)
            .write(&mut state.ircache_meta_buf)
            .dispatch([
                IRCACHE_CASCADE_SIZE as u32,
                IRCACHE_CASCADE_SIZE as u32,
                (IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_COUNT) as u32,
            ]);

            std::mem::swap(
                &mut state.ircache_grid_meta_buf,
                &mut state.ircache_grid_meta_buf2,
            );

            self.parity = (self.parity + 1) % 2;
        }

        /*SimpleRenderPass::new_compute(
            rg.add_pass("debug ircache"),
            "/shaders/ircache/ircache_draw_debug.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&gbuffer_depth.geometric_normal)
        .write(&mut state.ircache_meta_buf)
        .write(&mut state.ircache_grid_meta_buf)
        .write(&mut state.ircache_entry_cell_buf)
        .read(&state.ircache_spatial_buf)
        .read(&state.ircache_irradiance_buf)
        .write(&mut state.debug_out)
        .write(&mut state.ircache_pool_buf)
        .write(&mut state.ircache_life_buf)
        .write(&mut state.ircache_reposition_proposal_buf)
        .write(&mut state.ircache_reposition_proposal_count_buf)
        .constants(gbuffer_desc.extent_inv_extent_2d())
        .dispatch(gbuffer_desc.extent);

        state.draw_trace_origins(rg, self.debug_render_pass.clone(), gbuffer_depth);*/

        let indirect_args_buf = {
            let mut indirect_args_buf = rg.create(BufferDesc::new_gpu_only(
                (size_of::<u32>() * 4) * 2,
                vk::BufferUsageFlags::empty(),
            ));

            SimpleRenderPass::new_compute(
                rg.add_pass("_ircache dispatch args"),
                "/shaders/ircache/prepare_age_dispatch_args.hlsl",
            )
            .write(&mut state.ircache_meta_buf)
            .write(&mut indirect_args_buf)
            .dispatch([1, 1, 1]);

            indirect_args_buf
        };

        let mut entry_occupancy_buf = rg.create(BufferDesc::new_gpu_only(
            size_of::<u32>() * MAX_ENTRIES,
            vk::BufferUsageFlags::empty(),
        ));

        SimpleRenderPass::new_compute(
            rg.add_pass("age ircache entries"),
            "/shaders/ircache/age_ircache_entries.hlsl",
        )
        .write(&mut state.ircache_meta_buf)
        .write(&mut state.ircache_grid_meta_buf)
        .write(&mut state.ircache_entry_cell_buf)
        .write(&mut state.ircache_life_buf)
        .write(&mut state.ircache_pool_buf)
        .write(&mut state.ircache_spatial_buf)
        .write(&mut state.ircache_reposition_proposal_buf)
        .write(&mut state.ircache_reposition_proposal_count_buf)
        .write(&mut state.ircache_irradiance_buf)
        .write(&mut entry_occupancy_buf)
        .dispatch_indirect(&indirect_args_buf, 0);

        inclusive_prefix_scan_u32_1m(rg, &mut entry_occupancy_buf);

        SimpleRenderPass::new_compute(
            rg.add_pass("ircache compact"),
            "/shaders/ircache/ircache_compact_entries.hlsl",
        )
        .write(&mut state.ircache_meta_buf)
        .write(&mut state.ircache_life_buf)
        .read(&entry_occupancy_buf)
        .write(&mut state.ircache_entry_indirection_buf)
        .dispatch_indirect(&indirect_args_buf, 0);

        state
    }
}

pub struct IrcacheIrradiancePendingSummation {
    indirect_args_buf: rg::Handle<Buffer>,
}

impl IrcacheRenderState {
    #[allow(clippy::identity_op)]
    #[must_use]
    pub fn trace_irradiance(
        &mut self,
        rg: &mut rg::RenderGraph,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
        wrc: &WrcRenderState,
    ) -> IrcacheIrradiancePendingSummation {
        let indirect_args_buf = {
            let mut indirect_args_buf = rg.create(BufferDesc::new_gpu_only(
                (size_of::<u32>() * 4) * 4,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::INDIRECT_BUFFER,
            ));

            SimpleRenderPass::new_compute(
                rg.add_pass("_ircache dispatch args"),
                "/shaders/ircache/prepare_trace_dispatch_args.hlsl",
            )
            .read(&self.ircache_meta_buf)
            .write(&mut indirect_args_buf)
            .dispatch([1, 1, 1]);

            indirect_args_buf
        };

        SimpleRenderPass::new_compute(
            rg.add_pass("ircache reset"),
            "/shaders/ircache/reset_entry.hlsl",
        )
        .read(&self.ircache_life_buf)
        .read(&self.ircache_meta_buf)
        .read(&self.ircache_irradiance_buf)
        .write(&mut self.ircache_aux_buf)
        .read(&self.ircache_entry_indirection_buf)
        .dispatch_indirect(&indirect_args_buf, 16 * 2);

        SimpleRenderPass::new_rt(
            rg.add_pass("ircache trace access"),
            ShaderSource::hlsl("/shaders/ircache/trace_accessibility.rgen.hlsl"),
            [
                // Duplicated because `rt.hlsl` hardcodes miss index to 1
                ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
                ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
            ],
            std::iter::empty(),
        )
        .read(&self.ircache_spatial_buf)
        .read(&self.ircache_life_buf)
        .write_no_sync(&mut self.ircache_reposition_proposal_buf)
        // Only read-access necessary, but if we use `write_no_sync`, we can overlap
        // with the next pass. The next pass does not modify any data that this one reads.
        .write_no_sync(&mut self.ircache_meta_buf)
        .write_no_sync(&mut self.ircache_aux_buf)
        .read(&self.ircache_entry_indirection_buf)
        .trace_rays_indirect(tlas, &indirect_args_buf, 16 * 1);

        SimpleRenderPass::new_rt(
            rg.add_pass("ircache validate"),
            ShaderSource::hlsl("/shaders/ircache/ircache_validate.rgen.hlsl"),
            [
                ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
            ],
            [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
        )
        .read(&self.ircache_spatial_buf)
        .read(sky_cube)
        .write_no_sync(&mut self.ircache_grid_meta_buf)
        .read(&self.ircache_life_buf)
        .write_no_sync(&mut self.ircache_reposition_proposal_buf)
        .write_no_sync(&mut self.ircache_reposition_proposal_count_buf)
        .bind(wrc)
        .write_no_sync(&mut self.ircache_meta_buf)
        .write_no_sync(&mut self.ircache_aux_buf)
        .write_no_sync(&mut self.ircache_pool_buf)
        .read(&self.ircache_entry_indirection_buf)
        .write_no_sync(&mut self.ircache_entry_cell_buf)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(
            tlas,
            [
                (MAX_ENTRIES * IRCACHE_VALIDATION_SAMPLES_PER_FRAME) as u32,
                1,
                1,
            ],
        );
        // TODO: seems rather broken on AMD
        //.trace_rays_indirect(tlas, &indirect_args_buf, 16 * 3);

        SimpleRenderPass::new_rt(
            rg.add_pass("ircache trace"),
            ShaderSource::hlsl("/shaders/ircache/trace_irradiance.rgen.hlsl"),
            [
                ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
            ],
            [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
        )
        .read(&self.ircache_spatial_buf)
        .read(sky_cube)
        .write_no_sync(&mut self.ircache_grid_meta_buf)
        .read(&self.ircache_life_buf)
        .write_no_sync(&mut self.ircache_reposition_proposal_buf)
        .write_no_sync(&mut self.ircache_reposition_proposal_count_buf)
        .bind(wrc)
        .write_no_sync(&mut self.ircache_meta_buf)
        .write_no_sync(&mut self.ircache_aux_buf)
        .write_no_sync(&mut self.ircache_pool_buf)
        .read(&self.ircache_entry_indirection_buf)
        .write_no_sync(&mut self.ircache_entry_cell_buf)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(
            tlas,
            [(MAX_ENTRIES * IRCACHE_SAMPLES_PER_FRAME) as u32, 1, 1],
        );
        // TODO: seems rather broken on AMD
        //.trace_rays_indirect(tlas, &indirect_args_buf, 16 * 0);

        self.pending_irradiance_sum = true;

        IrcacheIrradiancePendingSummation { indirect_args_buf }
    }

    // Split from the RT passes, so that we don't immediately try to access the ray-traced data,
    // draining the GPU waves in the process.
    // TODO: doing this is a bit sketchy. Ideally the render graph would be flexible enough
    // to automatically allow re-scheduling of passes to hide latency.
    pub fn sum_up_irradiance_for_sampling(
        &mut self,
        rg: &mut rg::RenderGraph,
        pending: IrcacheIrradiancePendingSummation,
    ) {
        assert!(self.pending_irradiance_sum);

        SimpleRenderPass::new_compute(
            rg.add_pass("ircache sum"),
            "/shaders/ircache/sum_up_irradiance.hlsl",
        )
        .read(&self.ircache_life_buf)
        .write(&mut self.ircache_meta_buf)
        .write(&mut self.ircache_irradiance_buf)
        .write(&mut self.ircache_aux_buf)
        .read(&self.ircache_entry_indirection_buf)
        .dispatch_indirect(&pending.indirect_args_buf, 16 * 2);

        self.pending_irradiance_sum = false;
    }

    /*fn draw_trace_origins(
        &mut self,
        rg: &mut rg::RenderGraph,
        render_pass: Arc<RenderPass>,
        gbuffer_depth: &mut GbufferDepth,
    ) {
        let mut pass = rg.add_pass("raster ircache origins");

        let pipeline = pass.register_raster_pipeline(
            &[
                PipelineShaderDesc::builder(ShaderPipelineStage::Vertex)
                    .hlsl_source("/shaders/ircache/raster_origins_vs.hlsl")
                    .build()
                    .unwrap(),
                PipelineShaderDesc::builder(ShaderPipelineStage::Pixel)
                    .hlsl_source("/shaders/ircache/raster_origins_ps.hlsl")
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

        let ircache_meta_buf_ref = pass.read(
            &self.ircache_meta_buf,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let ircache_grid_meta_buf_ref = pass.read(
            &self.ircache_grid_meta_buf,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let ircache_life_buf_ref = pass.read(
            &self.ircache_life_buf,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let ircache_spatial_buf_ref = pass.read(
            &self.ircache_spatial_buf,
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
                    ircache_meta_buf_ref.bind(),
                    ircache_grid_meta_buf_ref.bind(),
                    ircache_life_buf_ref.bind(),
                    ircache_spatial_buf_ref.bind(),
                    //rg::RenderPassBinding::DynamicConstants(constants_offset),
                ],
            ));

            unsafe {
                let raw_device = &api.device().raw;
                let cb = api.cb;

                raw_device.cmd_draw(
                    cb.raw,
                    // 6 verts (two triangles) per cube face
                    6 * 6 * MAX_ENTRIES as u32,
                    1,
                    0,
                    0,
                );
            }

            api.end_render_pass();
        });
    }*/
}
