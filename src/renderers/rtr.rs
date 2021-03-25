use std::sync::Arc;

use kajiya_backend::{
    ash::vk,
    vk_sync,
    vulkan::{buffer::*, image::*, ray_tracing::RayTracingAcceleration},
    Device,
};
use kajiya_rg::{self as rg, GetOrCreateTemporal, SimpleRenderPass};

use super::{csgi2, GbufferDepth, PingPongTemporalResource};

use blue_noise_sampler::spp64::*;

pub struct RtrRenderer {
    temporal_tex: PingPongTemporalResource,
    temporal2_tex: PingPongTemporalResource,

    ranking_tile_buf: Arc<Buffer>,
    scambling_tile_buf: Arc<Buffer>,
    sobol_buf: Arc<Buffer>,
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
            ranking_tile_buf: make_lut_buffer(device, RANKING_TILE),
            scambling_tile_buf: make_lut_buffer(device, SCRAMBLING_TILE),
            sobol_buf: make_lut_buffer(device, SOBOL),
        }
    }
}

impl RtrRenderer {
    fn temporal_tex_desc(extent: [u32; 2]) -> ImageDesc {
        ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, extent)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
    }

    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
        csgi_volume: &csgi2::Csgi2Volume,
    ) -> rg::ReadOnlyHandle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let mut refl0_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let mut refl1_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

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
            "/assets/shaders/rtr/reflection.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&ranking_tile_buf)
        .read(&scambling_tile_buf)
        .read(&sobol_buf)
        .write(&mut refl0_tex)
        .write(&mut refl1_tex)
        .read(&csgi_volume.direct_cascade0)
        .read(&csgi_volume.indirect_cascade0)
        .read(sky_cube)
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

        let (mut temporal_output_tex, history_tex) = self
            .temporal_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(gbuffer_desc.extent_2d()));

        let mut ray_len_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R16_SFLOAT),
        );

        SimpleRenderPass::new_compute(
            rg.add_pass("reflection resolve"),
            "/assets/shaders/rtr/resolve.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&refl0_tex)
        .read(&refl1_tex)
        .read(&history_tex)
        .read(&reprojection_map)
        .read(&*half_view_normal_tex)
        .read(&*half_depth_tex)
        .write(&mut resolved_tex)
        .write(&mut ray_len_tex)
        .constants((
            resolved_tex.desc().extent_inv_extent_2d(),
            SPATIAL_RESOLVE_OFFSETS,
        ))
        .dispatch(resolved_tex.desc().extent);

        SimpleRenderPass::new_compute(
            rg.add_pass("reflection temporal"),
            "/assets/shaders/rtr/temporal_filter.hlsl",
        )
        .read(&resolved_tex)
        .read(&history_tex)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&ray_len_tex)
        .read(reprojection_map)
        .write(&mut temporal_output_tex)
        .constants(temporal_output_tex.desc().extent_inv_extent_2d())
        .dispatch(resolved_tex.desc().extent);

        let (mut temporal2_output_tex, history2_tex) = self
            .temporal2_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(gbuffer_desc.extent_2d()));

        SimpleRenderPass::new_compute(
            rg.add_pass("reflection temporal2"),
            "/assets/shaders/rtr/temporal_filter2.hlsl",
        )
        .read(&temporal_output_tex)
        .read(&history2_tex)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&ray_len_tex)
        .read(reprojection_map)
        .write(&mut temporal2_output_tex)
        .constants(temporal2_output_tex.desc().extent_inv_extent_2d())
        .dispatch(resolved_tex.desc().extent);

        temporal2_output_tex.into()
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
