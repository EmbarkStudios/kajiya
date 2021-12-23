use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration, shader::ShaderSource},
};
use kajiya_rg::{self as rg, SimpleRenderPass};

use super::{
    surfel_gi::SurfelGiRenderState, wrc::WrcRenderState, GbufferDepth, PingPongTemporalResource,
};

pub struct RtdgiRenderer {
    temporal_irradiance_tex: PingPongTemporalResource,
    temporal_ray_orig_tex: PingPongTemporalResource,
    temporal_ray_tex: PingPongTemporalResource,
    temporal_reservoir_tex: PingPongTemporalResource,
    temporal_candidate_tex: PingPongTemporalResource,

    temporal_validity_tex: PingPongTemporalResource,

    temporal2_tex: PingPongTemporalResource,
    temporal2_variance_tex: PingPongTemporalResource,
    temporal_hit_normal_tex: PingPongTemporalResource,

    pub spatial_reuse_pass_count: u32,
}

const COLOR_BUFFER_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

impl RtdgiRenderer {
    pub fn new() -> Self {
        Self {
            temporal_irradiance_tex: PingPongTemporalResource::new("rtdgi.irradiance"),
            temporal_ray_orig_tex: PingPongTemporalResource::new("rtdgi.ray_orig"),
            temporal_ray_tex: PingPongTemporalResource::new("rtdgi.ray"),
            temporal_reservoir_tex: PingPongTemporalResource::new("rtdgi.reservoir"),
            temporal_candidate_tex: PingPongTemporalResource::new("rtdgi.candidate"),
            temporal_validity_tex: PingPongTemporalResource::new("rtdgi.validity"),
            temporal2_tex: PingPongTemporalResource::new("rtdgi.temporal2"),
            temporal2_variance_tex: PingPongTemporalResource::new("rtdgi.temporal2_var"),
            temporal_hit_normal_tex: PingPongTemporalResource::new("rtdgi.hit_normal"),
            spatial_reuse_pass_count: 2,
        }
    }
}

impl RtdgiRenderer {
    fn temporal_tex_desc(extent: [u32; 2]) -> ImageDesc {
        ImageDesc::new_2d(COLOR_BUFFER_FORMAT, extent)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
    }

    #[allow(clippy::too_many_arguments)]
    fn temporal(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        input_color: &rg::Handle<Image>,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        reprojected_history_tex: &rg::Handle<Image>,
        rt_history_validity_tex: &rg::Handle<Image>,
        mut temporal_output_tex: rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let (mut temporal_variance_output_tex, variance_history_tex) =
            self.temporal2_variance_tex.get_output_and_history(
                rg,
                ImageDesc::new_2d(vk::Format::R16G16_SFLOAT, input_color.desc().extent_2d())
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            );

        let mut temporal_filtered_tex = rg.create(
            gbuffer_depth
                .gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi temporal"),
            "/shaders/rtdgi/temporal_filter2.hlsl",
        )
        .read(input_color)
        .read(reprojected_history_tex)
        .read(&variance_history_tex)
        .read(reprojection_map)
        .read(rt_history_validity_tex)
        .write(&mut temporal_filtered_tex)
        .write(&mut temporal_output_tex)
        .write(&mut temporal_variance_output_tex)
        .constants((
            temporal_output_tex.desc().extent_inv_extent_2d(),
            gbuffer_depth.gbuffer.desc().extent_inv_extent_2d(),
        ))
        .dispatch(temporal_output_tex.desc().extent);

        temporal_filtered_tex
    }

    fn spatial2(
        rg: &mut rg::TemporalRenderGraph,
        input_color: &rg::Handle<Image>,
        gbuffer_depth: &GbufferDepth,
        ssao_img: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
    ) -> rg::Handle<Image> {
        let mut spatial_filtered_tex =
            rg.create(Self::temporal_tex_desc(input_color.desc().extent_2d()));

        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi spatial2"),
            "/shaders/rtdgi/spatial_filter2.hlsl",
        )
        .read(input_color)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(ssao_img)
        .read(&gbuffer_depth.geometric_normal)
        .write(&mut spatial_filtered_tex)
        .constants((spatial_filtered_tex.desc().extent_inv_extent_2d(),))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .dispatch(spatial_filtered_tex.desc().extent);

        spatial_filtered_tex
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        surfel_gi: &SurfelGiRenderState,
        wrc: &WrcRenderState,
        tlas: &rg::Handle<RayTracingAcceleration>,
        ssao_img: &rg::Handle<Image>,
    ) -> rg::ReadOnlyHandle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let (temporal_output_tex, history_tex) = self
            .temporal2_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(gbuffer_desc.extent_2d()));

        let mut reprojected_history_tex =
            rg.create(Self::temporal_tex_desc(gbuffer_desc.extent_2d()));

        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi reproject"),
            "/shaders/rtdgi/fullres_reproject.hlsl",
        )
        .read(&history_tex)
        .read(reprojection_map)
        .write(&mut reprojected_history_tex)
        .constants((reprojected_history_tex.desc().extent_inv_extent_2d(),))
        .dispatch(reprojected_history_tex.desc().extent);

        let (mut hit_normal_output_tex, hit_normal_history_tex) =
            self.temporal_hit_normal_tex.get_output_and_history(
                rg,
                Self::temporal_tex_desc(
                    gbuffer_desc
                        // TODO: should really be rgba8
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .half_res()
                        .extent_2d(),
                ),
            );

        let (mut candidate_output_tex, candidate_history_tex) =
            self.temporal_candidate_tex.get_output_and_history(
                rg,
                ImageDesc::new_2d(
                    vk::Format::R16G16B16A16_SFLOAT,
                    gbuffer_desc.half_res().extent_2d(),
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            );

        let mut candidate_irradiance_tex = rg.create(
            gbuffer_desc
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let mut candidate_normal_tex = rg.create(
            gbuffer_desc
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let half_depth_tex = gbuffer_depth.half_depth(rg);

        let (mut validity_output_tex, validity_history_tex) =
            self.temporal_validity_tex.get_output_and_history(
                rg,
                Self::temporal_tex_desc(gbuffer_desc.half_res().extent_2d())
                    .format(vk::Format::R16_SFLOAT),
            );

        let (irradiance_tex, ray_tex, mut temporal_reservoir_tex) = {
            let (mut irradiance_output_tex, irradiance_history_tex) =
                self.temporal_irradiance_tex.get_output_and_history(
                    rg,
                    Self::temporal_tex_desc(gbuffer_desc.half_res().extent_2d()),
                );

            let (mut ray_orig_output_tex, ray_orig_history_tex) =
                self.temporal_ray_orig_tex.get_output_and_history(
                    rg,
                    ImageDesc::new_2d(
                        vk::Format::R32G32B32A32_SFLOAT,
                        gbuffer_desc.half_res().extent_2d(),
                    )
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                );

            let (mut reservoir_output_tex, reservoir_history_tex) =
                self.temporal_reservoir_tex.get_output_and_history(
                    rg,
                    ImageDesc::new_2d(
                        vk::Format::R32G32B32A32_SFLOAT,
                        gbuffer_desc.half_res().extent_2d(),
                    )
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                );

            let (mut ray_output_tex, ray_history_tex) =
                self.temporal_ray_tex.get_output_and_history(
                    rg,
                    Self::temporal_tex_desc(gbuffer_desc.half_res().extent_2d())
                        .format(vk::Format::R32G32B32A32_SFLOAT),
                );

            let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);

            let mut rt_history_validity_input_tex =
                rg.create(gbuffer_desc.half_res().format(vk::Format::R8_UNORM));

            SimpleRenderPass::new_rt(
                rg.add_pass("rtdgi trace"),
                ShaderSource::hlsl("/shaders/rtdgi/trace_diffuse.rgen.hlsl"),
                [
                    ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                    ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
                ],
                [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
            )
            .read(&*half_view_normal_tex)
            .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
            .read(&reprojected_history_tex)
            .read(&ray_history_tex)
            .read(ssao_img)
            .read(reprojection_map)
            .bind(surfel_gi)
            .bind(wrc)
            .read(sky_cube)
            .read(&ray_orig_history_tex)
            .write(&mut candidate_irradiance_tex)
            .write(&mut candidate_normal_tex)
            .write(&mut rt_history_validity_input_tex)
            .constants((gbuffer_desc.extent_inv_extent_2d(),))
            .raw_descriptor_set(1, bindless_descriptor_set)
            .trace_rays(tlas, candidate_irradiance_tex.desc().extent);

            SimpleRenderPass::new_compute(
                rg.add_pass("validity integrate"),
                "/shaders/rtdgi/temporal_validity_integrate.hlsl",
            )
            .read(&rt_history_validity_input_tex)
            .read(&validity_history_tex)
            .read(reprojection_map)
            .read(&*half_view_normal_tex)
            .read(&*half_depth_tex)
            .write(&mut validity_output_tex)
            .constants((
                gbuffer_desc.extent_inv_extent_2d(),
                validity_output_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(validity_output_tex.desc().extent);

            SimpleRenderPass::new_compute(
                rg.add_pass("restir temporal"),
                "/shaders/rtdgi/restir_temporal.hlsl",
            )
            .read(&*half_view_normal_tex)
            .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
            .read(&candidate_irradiance_tex)
            .read(&candidate_normal_tex)
            .read(&irradiance_history_tex)
            .read(&ray_orig_history_tex)
            .read(&ray_history_tex)
            .read(&reservoir_history_tex)
            .read(reprojection_map)
            .read(&hit_normal_history_tex)
            .read(&candidate_history_tex)
            .read(&validity_output_tex)
            .write(&mut irradiance_output_tex)
            .write(&mut ray_orig_output_tex)
            .write(&mut ray_output_tex)
            .write(&mut hit_normal_output_tex)
            .write(&mut reservoir_output_tex)
            .write(&mut candidate_output_tex)
            .constants((gbuffer_desc.extent_inv_extent_2d(),))
            .raw_descriptor_set(1, bindless_descriptor_set)
            .dispatch(irradiance_output_tex.desc().extent);

            (irradiance_output_tex, ray_output_tex, reservoir_output_tex)
        };

        let irradiance_tex = {
            let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);

            let mut irradiance_output_tex = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::empty())
                    .format(COLOR_BUFFER_FORMAT),
            );

            let mut reservoir_output_tex0 = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .half_res()
                    .format(vk::Format::R32G32B32A32_SFLOAT),
            );
            let mut reservoir_output_tex1 = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .half_res()
                    .format(vk::Format::R32G32B32A32_SFLOAT),
            );

            let mut reservoir_input_tex = &mut temporal_reservoir_tex;

            for spatial_reuse_pass_idx in 0..self.spatial_reuse_pass_count {
                SimpleRenderPass::new_compute(
                    rg.add_pass("restir spatial"),
                    "/shaders/rtdgi/restir_spatial.hlsl",
                )
                .read(&irradiance_tex)
                .read(&hit_normal_output_tex)
                .read(&ray_tex)
                .read(reservoir_input_tex)
                .read(&gbuffer_depth.gbuffer)
                .read(&*half_view_normal_tex)
                .read(&*half_depth_tex)
                .read(ssao_img)
                .read(&candidate_output_tex)
                .write(&mut reservoir_output_tex0)
                .constants((
                    gbuffer_desc.extent_inv_extent_2d(),
                    reservoir_output_tex0.desc().extent_inv_extent_2d(),
                    spatial_reuse_pass_idx as u32,
                ))
                .dispatch(reservoir_output_tex0.desc().extent);

                std::mem::swap(&mut reservoir_output_tex0, &mut reservoir_output_tex1);
                reservoir_input_tex = &mut reservoir_output_tex1;
            }

            SimpleRenderPass::new_compute(
                rg.add_pass("restir resolve"),
                "/shaders/rtdgi/restir_resolve.hlsl",
            )
            .read(&irradiance_tex)
            .read(&hit_normal_output_tex)
            .read(&ray_tex)
            .read(reservoir_input_tex)
            .read(&gbuffer_depth.gbuffer)
            .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
            .read(&*half_view_normal_tex)
            .read(&*half_depth_tex)
            .read(ssao_img)
            .read(&candidate_irradiance_tex)
            .read(&candidate_normal_tex)
            .write(&mut irradiance_output_tex)
            .raw_descriptor_set(1, bindless_descriptor_set)
            .constants((
                gbuffer_desc.extent_inv_extent_2d(),
                irradiance_output_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(irradiance_output_tex.desc().extent);

            irradiance_output_tex
        };

        /*let filtered_tex = self.temporal(
            rg,
            &irradiance_tex,
            gbuffer_depth,
            reprojection_map,
            sky_cube,
        );
        let filtered_tex = Self::spatial(rg, &filtered_tex, gbuffer_depth, ssao_img);

        let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);
        let half_depth_tex = gbuffer_depth.half_depth(rg);

        let mut upsampled_tex = rg.create(gbuffer_desc.format(vk::Format::R16G16B16A16_SFLOAT));
        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi upsample"),
            "/shaders/rtdgi/upsample.hlsl",
        )
        .read(&filtered_tex)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&gbuffer_depth.gbuffer)
        .read(&*half_view_normal_tex)
        .read(&*half_depth_tex)
        .read(ssao_img)
        .write(&mut upsampled_tex)
        .constants((
            upsampled_tex.desc().extent_inv_extent_2d(),
            super::rtr::SPATIAL_RESOLVE_OFFSETS,
        ))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .dispatch(upsampled_tex.desc().extent);*/
        let upsampled_tex = irradiance_tex;

        let filtered_tex = self.temporal(
            rg,
            &upsampled_tex,
            gbuffer_depth,
            reprojection_map,
            &reprojected_history_tex,
            &validity_output_tex,
            temporal_output_tex,
        );

        let filtered_tex = Self::spatial2(
            rg,
            &filtered_tex,
            gbuffer_depth,
            ssao_img,
            bindless_descriptor_set,
        );

        filtered_tex.into()
    }
}
