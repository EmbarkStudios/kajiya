use std::sync::Arc;

use kajiya_backend::{
    ash::vk,
    backend::{buffer::*, image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
    vk_sync, Device,
};
use rg::GetOrCreateTemporal;

use super::{csgi2, GbufferDepth, PingPongTemporalResource};

use blue_noise_sampler::spp16::*;

pub struct RtdgiRenderer {
    temporal_tex: PingPongTemporalResource,

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

impl RtdgiRenderer {
    pub fn new(device: &Device) -> Self {
        Self {
            temporal_tex: PingPongTemporalResource::new("rtdgi"),
            ranking_tile_buf: make_lut_buffer(device, RANKING_TILE),
            scambling_tile_buf: make_lut_buffer(device, SCRAMBLING_TILE),
            sobol_buf: make_lut_buffer(device, SOBOL),
        }
    }
}

impl RtdgiRenderer {
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

        // TODO: calculate specialized SSAO
        ssao_tex: &rg::Handle<Image>,
    ) -> rg::ReadOnlyHandle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let mut hit0_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let mut hit1_tex = rg.create(
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
            rg.add_pass("rtdgi trace"),
            "/assets/shaders/rtdgi/trace_diffuse.rgen.hlsl",
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
        .write(&mut hit0_tex)
        .write(&mut hit1_tex)
        .read(&csgi_volume.direct_cascade0)
        .read(&csgi_volume.indirect_cascade0)
        .read(sky_cube)
        .constants((gbuffer_desc.extent_inv_extent_2d(),))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, hit0_tex.desc().extent);

        let mut temporal_filtered_tex = rg.create(
            gbuffer_depth
                .gbuffer
                .desc()
                .half_res()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let mut spatial_filtered_tex = rg.create(*temporal_filtered_tex.desc());

        let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);
        let half_depth_tex = gbuffer_depth.half_depth(rg);

        let (mut temporal_output_tex, history_tex) = self.temporal_tex.get_output_and_history(
            rg,
            Self::temporal_tex_desc(spatial_filtered_tex.desc().extent_2d()),
        );

        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi temporal"),
            "/assets/shaders/rtdgi/temporal_filter.hlsl",
        )
        .read(&hit0_tex)
        .read(&history_tex)
        .read(reprojection_map)
        .read(&*half_view_normal_tex)
        .read(&*half_depth_tex)
        .read(&csgi_volume.direct_cascade0)
        .read(&csgi_volume.indirect_cascade0)
        .write(&mut temporal_output_tex)
        .write(&mut temporal_filtered_tex)
        .constants((
            temporal_output_tex.desc().extent_inv_extent_2d(),
            gbuffer_desc.extent_inv_extent_2d(),
        ))
        .dispatch(temporal_output_tex.desc().extent);

        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi spatial"),
            "/assets/shaders/rtdgi/spatial_filter.hlsl",
        )
        .read(&temporal_filtered_tex)
        .read(&hit1_tex)
        .read(&*half_view_normal_tex)
        .read(&*half_depth_tex)
        .read(ssao_tex)
        .write(&mut spatial_filtered_tex)
        .constants((
            spatial_filtered_tex.desc().extent_inv_extent_2d(),
            super::rtr::SPATIAL_RESOLVE_OFFSETS,
        ))
        .dispatch(spatial_filtered_tex.desc().extent);

        let mut upsampled_tex = rg.create(
            gbuffer_depth
                .gbuffer
                .desc()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );
        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi upsample"),
            "/assets/shaders/rtdgi/upsample.hlsl",
        )
        .read(&spatial_filtered_tex)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&gbuffer_depth.gbuffer)
        .read(&*half_view_normal_tex)
        .read(&*half_depth_tex)
        .read(ssao_tex)
        .write(&mut upsampled_tex)
        .constants((
            upsampled_tex.desc().extent_inv_extent_2d(),
            super::rtr::SPATIAL_RESOLVE_OFFSETS,
        ))
        .dispatch(upsampled_tex.desc().extent);

        upsampled_tex.into()
    }
}
