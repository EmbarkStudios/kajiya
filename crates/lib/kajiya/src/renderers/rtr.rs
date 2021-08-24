use std::sync::Arc;

use kajiya_backend::{
    ash::vk,
    vk_sync,
    vulkan::{buffer::*, image::*, ray_tracing::RayTracingAcceleration},
    Device,
};
use kajiya_rg::{self as rg, SimpleRenderPass};

use super::{csgi, GbufferDepth, PingPongTemporalResource};

use blue_noise_sampler::spp64::*;

pub struct RtrRenderer {
    temporal_tex: PingPongTemporalResource,
    temporal2_tex: PingPongTemporalResource,
    ray_len_tex: PingPongTemporalResource,

    ranking_tile_buf: Arc<Buffer>,
    scambling_tile_buf: Arc<Buffer>,
    sobol_buf: Arc<Buffer>,

    pub reservoir_resampling: bool,
}

fn as_byte_slice_unchecked<T: Copy>(v: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>())
    }
}

fn make_lut_buffer<T: Copy>(device: &Device, v: &[T]) -> Arc<Buffer> {
    Arc::new(
        device
            .create_buffer(
                BufferDesc::new(
                    v.len() * std::mem::size_of::<T>(),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                Some(as_byte_slice_unchecked(v)),
            )
            .unwrap(),
    )
}

impl RtrRenderer {
    pub fn new(device: &Device) -> Self {
        Self {
            temporal_tex: PingPongTemporalResource::new("rtr.temporal"),
            temporal2_tex: PingPongTemporalResource::new("rtr.temporal2"),
            ray_len_tex: PingPongTemporalResource::new("rtr.ray_len"),
            ranking_tile_buf: make_lut_buffer(device, RANKING_TILE),
            scambling_tile_buf: make_lut_buffer(device, SCRAMBLING_TILE),
            sobol_buf: make_lut_buffer(device, SOBOL),
            reservoir_resampling: false,
        }
    }
}

pub struct TracedRtr<'a> {
    pub resolved_tex: rg::Handle<Image>,
    temporal_output_tex: rg::Handle<Image>,
    history_tex: rg::Handle<Image>,
    ray_len_tex: rg::Handle<Image>,
    temporal2_tex: &'a mut PingPongTemporalResource,
}

impl RtrRenderer {
    fn temporal_tex_desc(extent: [u32; 2]) -> ImageDesc {
        ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, extent)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn trace(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
        csgi_volume: &csgi::CsgiVolume,
        rtdgi: &rg::Handle<Image>,
    ) -> TracedRtr {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let mut refl0_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        // When using PDFs stored wrt to the surface area metric, their values can be tiny or giant,
        // so fp32 is necessary. The projected solid angle metric is less sensitive, but that shader
        // variant is heavier. Overall the surface area metric and fp32 combo is faster on my RTX 2080.
        let mut refl1_tex = rg.create(refl0_tex.desc().format(vk::Format::R32G32B32A32_SFLOAT));

        let mut refl2_tex = rg.create(refl0_tex.desc().format(vk::Format::R8G8B8A8_SNORM));

        let ranking_tile_buf = rg.import(
            self.ranking_tile_buf.clone(),
            vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        );
        let scambling_tile_buf = rg.import(
            self.scambling_tile_buf.clone(),
            vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        );
        let sobol_buf = rg.import(
            self.sobol_buf.clone(),
            vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
        );

        SimpleRenderPass::new_rt(
            rg.add_pass("reflection trace"),
            "/shaders/rtr/reflection.rgen.hlsl",
            &[
                "/shaders/rt/gbuffer.rmiss.hlsl",
                "/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/shaders/rt/gbuffer.rchit.hlsl"],
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&ranking_tile_buf)
        .read(&scambling_tile_buf)
        .read(&sobol_buf)
        .read(rtdgi)
        .read(sky_cube)
        .read_array(&csgi_volume.indirect)
        .write(&mut refl0_tex)
        .write(&mut refl1_tex)
        .write(&mut refl2_tex)
        .constants((gbuffer_desc.extent_inv_extent_2d(),))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, refl0_tex.desc().extent);

        let mut resolved_tex = rg.create(
            gbuffer_depth
                .gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);
        let half_depth_tex = gbuffer_depth.half_depth(rg);

        let (temporal_output_tex, history_tex) = self
            .temporal_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(gbuffer_desc.extent_2d()));

        if self.reservoir_resampling {
            let mut refl0_exchanged_tex = rg.create(*refl0_tex.desc());
            let mut refl1_exchanged_tex = rg.create(*refl1_tex.desc());
            let mut refl2_exchanged_tex = rg.create(*refl2_tex.desc());

            SimpleRenderPass::new_compute(
                rg.add_pass("reflection exchange"),
                "/shaders/rtr/exchange.hlsl",
            )
            .read(&gbuffer_depth.gbuffer)
            .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
            .read(&refl0_tex)
            .read(&refl1_tex)
            .read(&refl2_tex)
            .read(&history_tex)
            .read(reprojection_map)
            .read(&*half_view_normal_tex)
            .read(&*half_depth_tex)
            .write(&mut refl0_exchanged_tex)
            .write(&mut refl1_exchanged_tex)
            .write(&mut refl2_exchanged_tex)
            .raw_descriptor_set(1, bindless_descriptor_set)
            .constants((
                refl0_exchanged_tex.desc().extent_inv_extent_2d(),
                SPATIAL_RESOLVE_OFFSETS,
            ))
            .dispatch(refl0_exchanged_tex.desc().extent);

            refl0_tex = refl0_exchanged_tex;
            refl1_tex = refl1_exchanged_tex;
            refl2_tex = refl2_exchanged_tex;
        }

        let (mut ray_len_output_tex, ray_len_history_tex) =
            self.ray_len_tex.get_output_and_history(
                rg,
                ImageDesc::new_2d(vk::Format::R16G16_SFLOAT, gbuffer_desc.extent_2d())
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            );

        SimpleRenderPass::new_compute(
            rg.add_pass("reflection resolve"),
            "/shaders/rtr/resolve.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&refl0_tex)
        .read(&refl1_tex)
        .read(&refl2_tex)
        .read(&history_tex)
        .read(reprojection_map)
        .read(&*half_view_normal_tex)
        .read(&*half_depth_tex)
        .read(&ray_len_history_tex)
        .write(&mut resolved_tex)
        .write(&mut ray_len_output_tex)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .constants((
            resolved_tex.desc().extent_inv_extent_2d(),
            SPATIAL_RESOLVE_OFFSETS,
        ))
        .dispatch(resolved_tex.desc().extent);

        TracedRtr {
            resolved_tex,
            temporal_output_tex,
            history_tex,
            ray_len_tex: ray_len_output_tex,
            temporal2_tex: &mut self.temporal2_tex,
        }
    }
}

impl<'a> TracedRtr<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn filter_temporal(
        mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        SimpleRenderPass::new_compute(
            rg.add_pass("reflection temporal"),
            "/shaders/rtr/temporal_filter.hlsl",
        )
        .read(&self.resolved_tex)
        .read(&self.history_tex)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&self.ray_len_tex)
        .read(reprojection_map)
        .write(&mut self.temporal_output_tex)
        .constants(self.temporal_output_tex.desc().extent_inv_extent_2d())
        .dispatch(self.resolved_tex.desc().extent);

        let (mut temporal2_output_tex, history2_tex) = self
            .temporal2_tex
            .get_output_and_history(rg, RtrRenderer::temporal_tex_desc(gbuffer_desc.extent_2d()));

        SimpleRenderPass::new_compute(
            rg.add_pass("reflection temporal2"),
            "/shaders/rtr/temporal_filter2.hlsl",
        )
        .read(&self.temporal_output_tex)
        .read(&history2_tex)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&self.ray_len_tex)
        .read(reprojection_map)
        .write(&mut temporal2_output_tex)
        .constants(temporal2_output_tex.desc().extent_inv_extent_2d())
        .dispatch(self.resolved_tex.desc().extent);

        SimpleRenderPass::new_compute(
            rg.add_pass("reflection cleanup"),
            "/shaders/rtr/spatial_cleanup.hlsl",
        )
        .read(&temporal2_output_tex)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&gbuffer_depth.geometric_normal)
        .write(&mut self.resolved_tex) // reuse
        .constants(SPATIAL_RESOLVE_OFFSETS)
        .dispatch(self.resolved_tex.desc().extent);

        self.resolved_tex
    }
}

pub const SPATIAL_RESOLVE_OFFSETS: [(i32, i32, i32, i32); 16 * 4 * 8] = [
    (0i32, 0i32, 0, 0),
    (-1i32, -1i32, 0, 0),
    (2i32, -1i32, 0, 0),
    (-2i32, -2i32, 0, 0),
    (-2i32, 2i32, 0, 0),
    (2i32, 2i32, 0, 0),
    (0i32, -3i32, 0, 0),
    (-3i32, 0i32, 0, 0),
    (3i32, 0i32, 0, 0),
    (3i32, 1i32, 0, 0),
    (-1i32, 3i32, 0, 0),
    (1i32, 3i32, 0, 0),
    (2i32, -3i32, 0, 0),
    (-3i32, -2i32, 0, 0),
    (1i32, 4i32, 0, 0),
    (-3i32, 3i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, 1i32, 0, 0),
    (-1i32, 1i32, 0, 0),
    (1i32, 1i32, 0, 0),
    (0i32, -2i32, 0, 0),
    (-2i32, 1i32, 0, 0),
    (1i32, 2i32, 0, 0),
    (2i32, -2i32, 0, 0),
    (1i32, -3i32, 0, 0),
    (-4i32, 0i32, 0, 0),
    (4i32, 0i32, 0, 0),
    (-1i32, -4i32, 0, 0),
    (-4i32, -1i32, 0, 0),
    (-4i32, 1i32, 0, 0),
    (-3i32, -3i32, 0, 0),
    (3i32, 3i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, -1i32, 0, 0),
    (-1i32, 0i32, 0, 0),
    (1i32, -1i32, 0, 0),
    (-2i32, -1i32, 0, 0),
    (2i32, 1i32, 0, 0),
    (-1i32, 2i32, 0, 0),
    (0i32, 3i32, 0, 0),
    (3i32, -1i32, 0, 0),
    (-2i32, -3i32, 0, 0),
    (3i32, -2i32, 0, 0),
    (-3i32, 2i32, 0, 0),
    (0i32, -4i32, 0, 0),
    (4i32, 1i32, 0, 0),
    (-1i32, 4i32, 0, 0),
    (3i32, -3i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (1i32, 0i32, 0, 0),
    (-2i32, 0i32, 0, 0),
    (2i32, 0i32, 0, 0),
    (0i32, 2i32, 0, 0),
    (-1i32, -2i32, 0, 0),
    (1i32, -2i32, 0, 0),
    (-1i32, -3i32, 0, 0),
    (-3i32, -1i32, 0, 0),
    (-3i32, 1i32, 0, 0),
    (3i32, 2i32, 0, 0),
    (-2i32, 3i32, 0, 0),
    (2i32, 3i32, 0, 0),
    (0i32, 4i32, 0, 0),
    (1i32, -4i32, 0, 0),
    (4i32, -1i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, 1i32, 0, 0),
    (-1i32, -1i32, 0, 0),
    (1i32, -2i32, 0, 0),
    (2i32, -2i32, 0, 0),
    (3i32, 0i32, 0, 0),
    (1i32, 3i32, 0, 0),
    (-3i32, 2i32, 0, 0),
    (-4i32, 0i32, 0, 0),
    (-1i32, -4i32, 0, 0),
    (-1i32, 4i32, 0, 0),
    (-3i32, -3i32, 0, 0),
    (4i32, 2i32, 0, 0),
    (1i32, -5i32, 0, 0),
    (-4i32, -4i32, 0, 0),
    (4i32, -4i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, -1i32, 0, 0),
    (1i32, 0i32, 0, 0),
    (-2i32, 1i32, 0, 0),
    (2i32, 2i32, 0, 0),
    (0i32, 3i32, 0, 0),
    (-3i32, -1i32, 0, 0),
    (-3i32, -2i32, 0, 0),
    (0i32, -4i32, 0, 0),
    (4i32, -1i32, 0, 0),
    (-3i32, 3i32, 0, 0),
    (3i32, 3i32, 0, 0),
    (-2i32, 4i32, 0, 0),
    (3i32, -4i32, 0, 0),
    (5i32, -1i32, 0, 0),
    (-2i32, -5i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-1i32, 1i32, 0, 0),
    (-1i32, -2i32, 0, 0),
    (2i32, 1i32, 0, 0),
    (0i32, -3i32, 0, 0),
    (3i32, -1i32, 0, 0),
    (-3i32, 1i32, 0, 0),
    (-1i32, 3i32, 0, 0),
    (3i32, -2i32, 0, 0),
    (1i32, 4i32, 0, 0),
    (3i32, -3i32, 0, 0),
    (-4i32, -2i32, 0, 0),
    (0i32, -5i32, 0, 0),
    (4i32, 3i32, 0, 0),
    (-5i32, -1i32, 0, 0),
    (5i32, 1i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-1i32, 0i32, 0, 0),
    (1i32, 1i32, 0, 0),
    (-2i32, 0i32, 0, 0),
    (2i32, -1i32, 0, 0),
    (1i32, 2i32, 0, 0),
    (-2i32, -2i32, 0, 0),
    (0i32, 4i32, 0, 0),
    (1i32, -4i32, 0, 0),
    (-4i32, 1i32, 0, 0),
    (4i32, 1i32, 0, 0),
    (-2i32, -4i32, 0, 0),
    (2i32, -4i32, 0, 0),
    (-4i32, 2i32, 0, 0),
    (-4i32, -3i32, 0, 0),
    (-3i32, 4i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, -1i32, 0, 0),
    (1i32, 1i32, 0, 0),
    (-2i32, -2i32, 0, 0),
    (2i32, -2i32, 0, 0),
    (-3i32, 0i32, 0, 0),
    (3i32, 1i32, 0, 0),
    (-1i32, 3i32, 0, 0),
    (-1i32, 4i32, 0, 0),
    (-2i32, -4i32, 0, 0),
    (5i32, 0i32, 0, 0),
    (-4i32, 3i32, 0, 0),
    (3i32, 4i32, 0, 0),
    (-5i32, -1i32, 0, 0),
    (2i32, -5i32, 0, 0),
    (4i32, -4i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (1i32, -1i32, 0, 0),
    (-2i32, 1i32, 0, 0),
    (1i32, 2i32, 0, 0),
    (1i32, -3i32, 0, 0),
    (-3i32, -2i32, 0, 0),
    (-3i32, 2i32, 0, 0),
    (4i32, 0i32, 0, 0),
    (4i32, 2i32, 0, 0),
    (2i32, 4i32, 0, 0),
    (3i32, -4i32, 0, 0),
    (-1i32, -5i32, 0, 0),
    (-5i32, -3i32, 0, 0),
    (-3i32, 5i32, 0, 0),
    (6i32, -2i32, 0, 0),
    (-6i32, 2i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, 1i32, 0, 0),
    (0i32, -2i32, 0, 0),
    (-1i32, -2i32, 0, 0),
    (-2i32, 2i32, 0, 0),
    (3i32, -1i32, 0, 0),
    (-4i32, 0i32, 0, 0),
    (1i32, 4i32, 0, 0),
    (3i32, 3i32, 0, 0),
    (4i32, -2i32, 0, 0),
    (0i32, -5i32, 0, 0),
    (-3i32, -4i32, 0, 0),
    (1i32, -5i32, 0, 0),
    (-5i32, 1i32, 0, 0),
    (-1i32, 5i32, 0, 0),
    (1i32, 5i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-1i32, 0i32, 0, 0),
    (0i32, 2i32, 0, 0),
    (2i32, 1i32, 0, 0),
    (0i32, -3i32, 0, 0),
    (-3i32, -1i32, 0, 0),
    (2i32, -3i32, 0, 0),
    (4i32, -1i32, 0, 0),
    (-3i32, -3i32, 0, 0),
    (-4i32, 2i32, 0, 0),
    (-2i32, 4i32, 0, 0),
    (5i32, 2i32, 0, 0),
    (4i32, 4i32, 0, 0),
    (5i32, -3i32, 0, 0),
    (-5i32, 3i32, 0, 0),
    (3i32, 5i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, 1i32, 0, 0),
    (1i32, -1i32, 0, 0),
    (0i32, -3i32, 0, 0),
    (-3i32, 0i32, 0, 0),
    (3i32, 1i32, 0, 0),
    (0i32, 4i32, 0, 0),
    (-4i32, -3i32, 0, 0),
    (-4i32, 3i32, 0, 0),
    (4i32, 3i32, 0, 0),
    (5i32, -1i32, 0, 0),
    (-3i32, -5i32, 0, 0),
    (3i32, -5i32, 0, 0),
    (5i32, -3i32, 0, 0),
    (-3i32, 5i32, 0, 0),
    (-6i32, 0i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-2i32, -1i32, 0, 0),
    (2i32, -1i32, 0, 0),
    (-2i32, -2i32, 0, 0),
    (-2i32, 2i32, 0, 0),
    (1i32, 3i32, 0, 0),
    (-4i32, 1i32, 0, 0),
    (4i32, 1i32, 0, 0),
    (3i32, -3i32, 0, 0),
    (-1i32, -5i32, 0, 0),
    (2i32, -5i32, 0, 0),
    (-2i32, 5i32, 0, 0),
    (2i32, 5i32, 0, 0),
    (0i32, -6i32, 0, 0),
    (-6i32, -1i32, 0, 0),
    (6i32, 2i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, 2i32, 0, 0),
    (-2i32, 1i32, 0, 0),
    (1i32, -3i32, 0, 0),
    (3i32, -1i32, 0, 0),
    (3i32, 2i32, 0, 0),
    (-1i32, -4i32, 0, 0),
    (-4i32, -2i32, 0, 0),
    (-2i32, 4i32, 0, 0),
    (5i32, -2i32, 0, 0),
    (-4i32, -4i32, 0, 0),
    (3i32, 5i32, 0, 0),
    (0i32, 6i32, 0, 0),
    (-6i32, -2i32, 0, 0),
    (-6i32, 2i32, 0, 0),
    (4i32, -5i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, -1i32, 0, 0),
    (1i32, 1i32, 0, 0),
    (-1i32, -2i32, 0, 0),
    (-2i32, 3i32, 0, 0),
    (1i32, 4i32, 0, 0),
    (-3i32, -4i32, 0, 0),
    (4i32, -3i32, 0, 0),
    (5i32, 0i32, 0, 0),
    (1i32, -5i32, 0, 0),
    (-5i32, -1i32, 0, 0),
    (-5i32, 2i32, 0, 0),
    (5i32, 3i32, 0, 0),
    (6i32, 1i32, 0, 0),
    (1i32, 6i32, 0, 0),
    (-4i32, 6i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (1i32, -3i32, 0, 0),
    (1i32, 3i32, 0, 0),
    (-3i32, 2i32, 0, 0),
    (4i32, -1i32, 0, 0),
    (3i32, -4i32, 0, 0),
    (-5i32, 0i32, 0, 0),
    (-1i32, -5i32, 0, 0),
    (5i32, 2i32, 0, 0),
    (-2i32, 5i32, 0, 0),
    (-4i32, -4i32, 0, 0),
    (1i32, 6i32, 0, 0),
    (-5i32, 5i32, 0, 0),
    (3i32, 7i32, 0, 0),
    (-8i32, 1i32, 0, 0),
    (7i32, 4i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-1i32, -1i32, 0, 0),
    (2i32, -1i32, 0, 0),
    (-1i32, 2i32, 0, 0),
    (3i32, 1i32, 0, 0),
    (-4i32, -2i32, 0, 0),
    (2i32, 4i32, 0, 0),
    (-2i32, -5i32, 0, 0),
    (-5i32, 2i32, 0, 0),
    (6i32, -2i32, 0, 0),
    (-2i32, 6i32, 0, 0),
    (5i32, -4i32, 0, 0),
    (6i32, 3i32, 0, 0),
    (0i32, -7i32, 0, 0),
    (5i32, 5i32, 0, 0),
    (4i32, -6i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (2i32, 2i32, 0, 0),
    (-1i32, -3i32, 0, 0),
    (-3i32, -1i32, 0, 0),
    (0i32, 4i32, 0, 0),
    (-3i32, 3i32, 0, 0),
    (4i32, -2i32, 0, 0),
    (1i32, -5i32, 0, 0),
    (-5i32, -3i32, 0, 0),
    (6i32, 0i32, 0, 0),
    (-6i32, 1i32, 0, 0),
    (3i32, -6i32, 0, 0),
    (3i32, 6i32, 0, 0),
    (-1i32, 7i32, 0, 0),
    (-4i32, -7i32, 0, 0),
    (-7i32, -4i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-1i32, 0i32, 0, 0),
    (0i32, 2i32, 0, 0),
    (-2i32, -2i32, 0, 0),
    (2i32, -2i32, 0, 0),
    (4i32, 1i32, 0, 0),
    (-2i32, 4i32, 0, 0),
    (4i32, 3i32, 0, 0),
    (-3i32, -5i32, 0, 0),
    (5i32, -3i32, 0, 0),
    (-6i32, 0i32, 0, 0),
    (1i32, -6i32, 0, 0),
    (-5i32, 4i32, 0, 0),
    (7i32, 0i32, 0, 0),
    (1i32, 7i32, 0, 0),
    (-3i32, -7i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, -1i32, 0, 0),
    (-2i32, 2i32, 0, 0),
    (3i32, 2i32, 0, 0),
    (-3i32, -3i32, 0, 0),
    (-5i32, 2i32, 0, 0),
    (0i32, -6i32, 0, 0),
    (6i32, 1i32, 0, 0),
    (-1i32, 6i32, 0, 0),
    (-6i32, -2i32, 0, 0),
    (6i32, -3i32, 0, 0),
    (3i32, 6i32, 0, 0),
    (4i32, -6i32, 0, 0),
    (-5i32, 6i32, 0, 0),
    (7i32, 4i32, 0, 0),
    (-8i32, 2i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (1i32, 2i32, 0, 0),
    (1i32, -3i32, 0, 0),
    (-4i32, 0i32, 0, 0),
    (4i32, -1i32, 0, 0),
    (-3i32, 4i32, 0, 0),
    (3i32, -5i32, 0, 0),
    (1i32, 6i32, 0, 0),
    (-2i32, -6i32, 0, 0),
    (-5i32, -5i32, 0, 0),
    (-8i32, 0i32, 0, 0),
    (8i32, -1i32, 0, 0),
    (4i32, 7i32, 0, 0),
    (7i32, -5i32, 0, 0),
    (2i32, -9i32, 0, 0),
    (-7i32, -6i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-1i32, 0i32, 0, 0),
    (-2i32, -2i32, 0, 0),
    (0i32, 3i32, 0, 0),
    (3i32, -2i32, 0, 0),
    (-1i32, -4i32, 0, 0),
    (4i32, 3i32, 0, 0),
    (5i32, -2i32, 0, 0),
    (1i32, -6i32, 0, 0),
    (-6i32, 1i32, 0, 0),
    (-2i32, 6i32, 0, 0),
    (5i32, 4i32, 0, 0),
    (-6i32, -3i32, 0, 0),
    (-6i32, 4i32, 0, 0),
    (8i32, 0i32, 0, 0),
    (-4i32, -7i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (2i32, 0i32, 0, 0),
    (-2i32, 1i32, 0, 0),
    (-4i32, -1i32, 0, 0),
    (1i32, 4i32, 0, 0),
    (3i32, -4i32, 0, 0),
    (5i32, 1i32, 0, 0),
    (-4i32, 4i32, 0, 0),
    (3i32, 5i32, 0, 0),
    (-4i32, -5i32, 0, 0),
    (-7i32, -2i32, 0, 0),
    (7i32, -2i32, 0, 0),
    (2i32, -8i32, 0, 0),
    (8i32, 3i32, 0, 0),
    (-3i32, 8i32, 0, 0),
    (5i32, -7i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, -1i32, 0, 0),
    (4i32, -3i32, 0, 0),
    (-4i32, 3i32, 0, 0),
    (0i32, 5i32, 0, 0),
    (4i32, 4i32, 0, 0),
    (-3i32, -5i32, 0, 0),
    (1i32, -6i32, 0, 0),
    (-6i32, -2i32, 0, 0),
    (7i32, -1i32, 0, 0),
    (0i32, 8i32, 0, 0),
    (-8i32, 3i32, 0, 0),
    (8i32, 3i32, 0, 0),
    (5i32, 7i32, 0, 0),
    (7i32, -6i32, 0, 0),
    (-5i32, 8i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (2i32, 2i32, 0, 0),
    (0i32, -3i32, 0, 0),
    (3i32, 0i32, 0, 0),
    (-3i32, -1i32, 0, 0),
    (-1i32, 3i32, 0, 0),
    (-6i32, 1i32, 0, 0),
    (4i32, -5i32, 0, 0),
    (6i32, 3i32, 0, 0),
    (-6i32, -5i32, 0, 0),
    (-6i32, 5i32, 0, 0),
    (0i32, -8i32, 0, 0),
    (-9i32, -1i32, 0, 0),
    (-2i32, 9i32, 0, 0),
    (5i32, -8i32, 0, 0),
    (-3i32, -9i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-2i32, 1i32, 0, 0),
    (-1i32, -4i32, 0, 0),
    (5i32, 0i32, 0, 0),
    (2i32, 6i32, 0, 0),
    (-3i32, 6i32, 0, 0),
    (-4i32, -6i32, 0, 0),
    (3i32, -7i32, 0, 0),
    (-7i32, -3i32, 0, 0),
    (7i32, -3i32, 0, 0),
    (8i32, 1i32, 0, 0),
    (8i32, 6i32, 0, 0),
    (-10i32, 1i32, 0, 0),
    (-7i32, 8i32, 0, 0),
    (-10i32, 4i32, 0, 0),
    (0i32, 11i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (1i32, 2i32, 0, 0),
    (2i32, -2i32, 0, 0),
    (-3i32, -3i32, 0, 0),
    (-3i32, 3i32, 0, 0),
    (3i32, -4i32, 0, 0),
    (-5i32, 0i32, 0, 0),
    (3i32, 4i32, 0, 0),
    (-2i32, 6i32, 0, 0),
    (-1i32, -7i32, 0, 0),
    (6i32, 5i32, 0, 0),
    (-8i32, 2i32, 0, 0),
    (2i32, 8i32, 0, 0),
    (7i32, -5i32, 0, 0),
    (-7i32, 5i32, 0, 0),
    (2i32, -9i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (0i32, -2i32, 0, 0),
    (2i32, 2i32, 0, 0),
    (-4i32, -2i32, 0, 0),
    (-2i32, 5i32, 0, 0),
    (-7i32, 1i32, 0, 0),
    (4i32, -7i32, 0, 0),
    (7i32, -4i32, 0, 0),
    (6i32, 7i32, 0, 0),
    (9i32, 3i32, 0, 0),
    (-6i32, -9i32, 0, 0),
    (-9i32, 6i32, 0, 0),
    (0i32, -11i32, 0, 0),
    (0i32, 11i32, 0, 0),
    (-10i32, -6i32, 0, 0),
    (-7i32, 10i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-3i32, 1i32, 0, 0),
    (3i32, -2i32, 0, 0),
    (1i32, -7i32, 0, 0),
    (-6i32, 4i32, 0, 0),
    (2i32, 7i32, 0, 0),
    (-7i32, -4i32, 0, 0),
    (8i32, -1i32, 0, 0),
    (-2i32, -8i32, 0, 0),
    (-10i32, 3i32, 0, 0),
    (10i32, -4i32, 0, 0),
    (-4i32, 10i32, 0, 0),
    (4i32, 10i32, 0, 0),
    (-11i32, -2i32, 0, 0),
    (12i32, 1i32, 0, 0),
    (10i32, 7i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (-1i32, 2i32, 0, 0),
    (-1i32, -4i32, 0, 0),
    (4i32, 1i32, 0, 0),
    (4i32, -4i32, 0, 0),
    (4i32, 5i32, 0, 0),
    (-5i32, -5i32, 0, 0),
    (7i32, 3i32, 0, 0),
    (-3i32, 7i32, 0, 0),
    (-8i32, -1i32, 0, 0),
    (-6i32, 6i32, 0, 0),
    (0i32, 9i32, 0, 0),
    (3i32, -10i32, 0, 0),
    (7i32, -8i32, 0, 0),
    (-9i32, -8i32, 0, 0),
    (-12i32, 3i32, 0, 0),
    (0i32, 0i32, 0, 0),
    (1i32, -5i32, 0, 0),
    (-5i32, 1i32, 0, 0),
    (1i32, 5i32, 0, 0),
    (-3i32, -5i32, 0, 0),
    (6i32, 0i32, 0, 0),
    (-4i32, -8i32, 0, 0),
    (-10i32, 0i32, 0, 0),
    (10i32, -1i32, 0, 0),
    (5i32, 9i32, 0, 0),
    (8i32, -7i32, 0, 0),
    (9i32, 6i32, 0, 0),
    (-11i32, -4i32, 0, 0),
    (2i32, 12i32, 0, 0),
    (-7i32, -10i32, 0, 0),
    (-4i32, 12i32, 0, 0),
];
