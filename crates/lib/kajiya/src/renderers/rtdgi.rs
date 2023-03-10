use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration, shader::ShaderSource},
};
use kajiya_rg::{self as rg, SimpleRenderPass};

use super::{
    ircache::IrcacheRenderState, wrc::WrcRenderState, GbufferDepth, PingPongTemporalResource,
};

pub struct RtdgiRenderer {
    temporal_radiance_tex: PingPongTemporalResource,
    temporal_ray_orig_tex: PingPongTemporalResource,
    temporal_ray_tex: PingPongTemporalResource,
    temporal_reservoir_tex: PingPongTemporalResource,
    temporal_candidate_tex: PingPongTemporalResource,

    temporal_invalidity_tex: PingPongTemporalResource,

    temporal2_tex: PingPongTemporalResource,
    temporal2_variance_tex: PingPongTemporalResource,
    temporal_hit_normal_tex: PingPongTemporalResource,

    pub spatial_reuse_pass_count: u32,
    pub use_raytraced_reservoir_visibility: bool,
}

const COLOR_BUFFER_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

impl Default for RtdgiRenderer {
    fn default() -> Self {
        Self {
            temporal_radiance_tex: PingPongTemporalResource::new("rtdgi.radiance"),
            temporal_ray_orig_tex: PingPongTemporalResource::new("rtdgi.ray_orig"),
            temporal_ray_tex: PingPongTemporalResource::new("rtdgi.ray"),
            temporal_reservoir_tex: PingPongTemporalResource::new("rtdgi.reservoir"),
            temporal_candidate_tex: PingPongTemporalResource::new("rtdgi.candidate"),
            temporal_invalidity_tex: PingPongTemporalResource::new("rtdgi.invalidity"),
            temporal2_tex: PingPongTemporalResource::new("rtdgi.temporal2"),
            temporal2_variance_tex: PingPongTemporalResource::new("rtdgi.temporal2_var"),
            temporal_hit_normal_tex: PingPongTemporalResource::new("rtdgi.hit_normal"),
            spatial_reuse_pass_count: 2,
            use_raytraced_reservoir_visibility: false,
        }
    }
}

pub struct ReprojectedRtdgi {
    reprojected_history_tex: rg::Handle<Image>,
    temporal_output_tex: rg::Handle<Image>,
}

pub struct RtdgiCandidates {
    pub candidate_radiance_tex: rg::Handle<Image>,
    pub candidate_normal_tex: rg::Handle<Image>,
    pub candidate_hit_tex: rg::Handle<Image>,
}

pub struct RtdgiOutput {
    pub screen_irradiance_tex: rg::ReadOnlyHandle<Image>,
    pub candidates: RtdgiCandidates,
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
        rt_history_invalidity_tex: &rg::Handle<Image>,
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
            "/shaders/rtdgi/temporal_filter.hlsl",
        )
        .read(input_color)
        .read(reprojected_history_tex)
        .read(&variance_history_tex)
        .read(reprojection_map)
        .read(rt_history_invalidity_tex)
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

    fn spatial(
        rg: &mut rg::TemporalRenderGraph,
        input_color: &rg::Handle<Image>,
        gbuffer_depth: &GbufferDepth,
        ssao_tex: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
    ) -> rg::Handle<Image> {
        let mut spatial_filtered_tex =
            rg.create(Self::temporal_tex_desc(input_color.desc().extent_2d()));

        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi spatial"),
            "/shaders/rtdgi/spatial_filter.hlsl",
        )
        .read(input_color)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(ssao_tex)
        .read(&gbuffer_depth.geometric_normal)
        .write(&mut spatial_filtered_tex)
        .constants((spatial_filtered_tex.desc().extent_inv_extent_2d(),))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .dispatch(spatial_filtered_tex.desc().extent);

        spatial_filtered_tex
    }

    pub fn reproject(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        reprojection_map: &rg::Handle<Image>,
    ) -> ReprojectedRtdgi {
        let gbuffer_extent = reprojection_map.desc().extent_2d();

        let (temporal_output_tex, history_tex) = self
            .temporal2_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(gbuffer_extent));

        let mut reprojected_history_tex = rg.create(Self::temporal_tex_desc(gbuffer_extent));

        SimpleRenderPass::new_compute(
            rg.add_pass("rtdgi reproject"),
            "/shaders/rtdgi/fullres_reproject.hlsl",
        )
        .read(&history_tex)
        .read(reprojection_map)
        .write(&mut reprojected_history_tex)
        .constants((reprojected_history_tex.desc().extent_inv_extent_2d(),))
        .dispatch(reprojected_history_tex.desc().extent);

        ReprojectedRtdgi {
            reprojected_history_tex,
            temporal_output_tex,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        ReprojectedRtdgi {
            reprojected_history_tex,
            temporal_output_tex,
        }: ReprojectedRtdgi,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        ircache: &mut IrcacheRenderState,
        wrc: &WrcRenderState,
        tlas: &rg::Handle<RayTracingAcceleration>,
        ssao_tex: &rg::Handle<Image>,
    ) -> RtdgiOutput {
        let mut half_ssao_tex = rg.create(
            ssao_tex
                .desc()
                .half_res()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R8_SNORM),
        );
        SimpleRenderPass::new_compute(
            rg.add_pass("extract ssao/2"),
            "/shaders/extract_half_res_ssao.hlsl",
        )
        .read(ssao_tex)
        .write(&mut half_ssao_tex)
        .dispatch(half_ssao_tex.desc().extent);

        let gbuffer_desc = gbuffer_depth.gbuffer.desc();

        let (mut hit_normal_output_tex, hit_normal_history_tex) =
            self.temporal_hit_normal_tex.get_output_and_history(
                rg,
                Self::temporal_tex_desc(
                    gbuffer_desc
                        .format(vk::Format::R8G8B8A8_UNORM)
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

        let mut candidate_radiance_tex = rg.create(
            gbuffer_desc
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let mut candidate_normal_tex =
            rg.create(gbuffer_desc.half_res().format(vk::Format::R8G8B8A8_SNORM));

        let mut candidate_hit_tex = rg.create(
            gbuffer_desc
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let mut temporal_reservoir_packed_tex = rg.create(
            gbuffer_desc
                .half_res()
                .format(vk::Format::R32G32B32A32_UINT),
        );

        let half_depth_tex = gbuffer_depth.half_depth(rg);

        let (mut invalidity_output_tex, invalidity_history_tex) =
            self.temporal_invalidity_tex.get_output_and_history(
                rg,
                Self::temporal_tex_desc(gbuffer_desc.half_res().extent_2d())
                    .format(vk::Format::R16G16_SFLOAT),
            );

        let (radiance_tex, mut temporal_reservoir_tex) = {
            let (mut radiance_output_tex, mut radiance_history_tex) =
                self.temporal_radiance_tex.get_output_and_history(
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

            let (mut ray_output_tex, ray_history_tex) =
                self.temporal_ray_tex.get_output_and_history(
                    rg,
                    Self::temporal_tex_desc(gbuffer_desc.half_res().extent_2d())
                        .format(vk::Format::R16G16B16A16_SFLOAT),
                );

            let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);

            let mut rt_history_validity_pre_input_tex =
                rg.create(gbuffer_desc.half_res().format(vk::Format::R8_UNORM));

            let (mut reservoir_output_tex, mut reservoir_history_tex) =
                self.temporal_reservoir_tex.get_output_and_history(
                    rg,
                    ImageDesc::new_2d(vk::Format::R32G32_UINT, gbuffer_desc.half_res().extent_2d())
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                );

            SimpleRenderPass::new_rt(
                rg.add_pass("rtdgi validate"),
                ShaderSource::hlsl("/shaders/rtdgi/diffuse_validate.rgen.hlsl"),
                [
                    ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                    ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
                ],
                [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
            )
            .read(&*half_view_normal_tex)
            .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
            .read(&reprojected_history_tex)
            .write(&mut reservoir_history_tex)
            .read(&ray_history_tex)
            .read(reprojection_map)
            .bind_mut(ircache)
            .bind(wrc)
            .read(sky_cube)
            .write(&mut radiance_history_tex)
            .read(&ray_orig_history_tex)
            .write(&mut rt_history_validity_pre_input_tex)
            .constants((gbuffer_desc.extent_inv_extent_2d(),))
            .raw_descriptor_set(1, bindless_descriptor_set)
            .trace_rays(tlas, candidate_radiance_tex.desc().extent);

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
            .read(reprojection_map)
            .bind_mut(ircache)
            .bind(wrc)
            .read(sky_cube)
            .read(&ray_orig_history_tex)
            .write(&mut candidate_radiance_tex)
            .write(&mut candidate_normal_tex)
            .write(&mut candidate_hit_tex)
            .read(&rt_history_validity_pre_input_tex)
            .write(&mut rt_history_validity_input_tex)
            .constants((gbuffer_desc.extent_inv_extent_2d(),))
            .raw_descriptor_set(1, bindless_descriptor_set)
            .trace_rays(tlas, candidate_radiance_tex.desc().extent);

            SimpleRenderPass::new_compute(
                rg.add_pass("validity integrate"),
                "/shaders/rtdgi/temporal_validity_integrate.hlsl",
            )
            .read(&rt_history_validity_input_tex)
            .read(&invalidity_history_tex)
            .read(reprojection_map)
            .read(&*half_view_normal_tex)
            .read(&*half_depth_tex)
            .write(&mut invalidity_output_tex)
            .constants((
                gbuffer_desc.extent_inv_extent_2d(),
                invalidity_output_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(invalidity_output_tex.desc().extent);

            SimpleRenderPass::new_compute(
                rg.add_pass("restir temporal"),
                "/shaders/rtdgi/restir_temporal.hlsl",
            )
            .read(&*half_view_normal_tex)
            .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
            .read(&candidate_radiance_tex)
            .read(&candidate_normal_tex)
            .read(&candidate_hit_tex)
            .read(&radiance_history_tex)
            .read(&ray_orig_history_tex)
            .read(&ray_history_tex)
            .read(&reservoir_history_tex)
            .read(reprojection_map)
            .read(&hit_normal_history_tex)
            .read(&candidate_history_tex)
            .read(&invalidity_output_tex)
            .write(&mut radiance_output_tex)
            .write(&mut ray_orig_output_tex)
            .write(&mut ray_output_tex)
            .write(&mut hit_normal_output_tex)
            .write(&mut reservoir_output_tex)
            .write(&mut candidate_output_tex)
            .write(&mut temporal_reservoir_packed_tex)
            .constants((gbuffer_desc.extent_inv_extent_2d(),))
            .raw_descriptor_set(1, bindless_descriptor_set)
            .dispatch(radiance_output_tex.desc().extent);

            (radiance_output_tex, reservoir_output_tex)
        };

        let irradiance_tex = {
            let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);

            let mut reservoir_output_tex0 = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .half_res()
                    .format(vk::Format::R32G32_UINT),
            );
            let mut reservoir_output_tex1 = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .half_res()
                    .format(vk::Format::R32G32_UINT),
            );

            // Note: only needed with `RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE`
            // Consider making that a CPU-side setting too.
            let mut bounced_radiance_output_tex0 = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .half_res()
                    .format(vk::Format::B10G11R11_UFLOAT_PACK32),
            );
            let mut bounced_radiance_output_tex1 = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
                    .half_res()
                    .format(vk::Format::B10G11R11_UFLOAT_PACK32),
            );

            let mut reservoir_input_tex = &mut temporal_reservoir_tex;
            let mut bounced_radiance_input_tex = &radiance_tex;

            for spatial_reuse_pass_idx in 0..self.spatial_reuse_pass_count {
                // Only do occlusion checks in the final resampling pass.
                // Otherwise we get accumulation of darkening.
                let perform_occulsion_raymarch: u32 =
                    if spatial_reuse_pass_idx + 1 == self.spatial_reuse_pass_count {
                        1
                    } else {
                        0
                    };

                let occlusion_raymarch_importance_only: u32 =
                    if self.use_raytraced_reservoir_visibility {
                        1
                    } else {
                        0
                    };

                SimpleRenderPass::new_compute(
                    rg.add_pass("restir spatial"),
                    "/shaders/rtdgi/restir_spatial.hlsl",
                )
                .read(reservoir_input_tex)
                .read(bounced_radiance_input_tex)
                .read(&*half_view_normal_tex)
                .read(&*half_depth_tex)
                .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
                .read(&half_ssao_tex)
                .read(&temporal_reservoir_packed_tex)
                .read(&reprojected_history_tex)
                .write(&mut reservoir_output_tex0)
                .write(&mut bounced_radiance_output_tex0)
                .constants((
                    gbuffer_desc.extent_inv_extent_2d(),
                    reservoir_output_tex0.desc().extent_inv_extent_2d(),
                    spatial_reuse_pass_idx,
                    perform_occulsion_raymarch,
                    occlusion_raymarch_importance_only,
                ))
                .dispatch(reservoir_output_tex0.desc().extent);

                std::mem::swap(&mut reservoir_output_tex0, &mut reservoir_output_tex1);
                std::mem::swap(
                    &mut bounced_radiance_output_tex0,
                    &mut bounced_radiance_output_tex1,
                );

                reservoir_input_tex = &mut reservoir_output_tex1;
                bounced_radiance_input_tex = &mut bounced_radiance_output_tex1;
            }

            if self.use_raytraced_reservoir_visibility {
                SimpleRenderPass::new_rt(
                    rg.add_pass("restir check"),
                    ShaderSource::hlsl("/shaders/rtdgi/restir_check.rgen.hlsl"),
                    [
                        ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                        ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
                    ],
                    [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
                )
                .read(&*half_depth_tex)
                .read(&temporal_reservoir_packed_tex)
                .write(reservoir_input_tex)
                .constants((gbuffer_desc.extent_inv_extent_2d(),))
                .raw_descriptor_set(1, bindless_descriptor_set)
                .trace_rays(tlas, candidate_radiance_tex.desc().extent);
            }

            let mut irradiance_output_tex = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::empty())
                    .format(COLOR_BUFFER_FORMAT),
            );

            SimpleRenderPass::new_compute(
                rg.add_pass("restir resolve"),
                "/shaders/rtdgi/restir_resolve.hlsl",
            )
            .read(&radiance_tex)
            .read(reservoir_input_tex)
            .read(&gbuffer_depth.gbuffer)
            .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
            .read(&*half_view_normal_tex)
            .read(&*half_depth_tex)
            .read(ssao_tex)
            .read(&candidate_radiance_tex)
            .read(&candidate_hit_tex)
            .read(&temporal_reservoir_packed_tex)
            .read(bounced_radiance_input_tex)
            .write(&mut irradiance_output_tex)
            .raw_descriptor_set(1, bindless_descriptor_set)
            .constants((
                gbuffer_desc.extent_inv_extent_2d(),
                irradiance_output_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(irradiance_output_tex.desc().extent);

            irradiance_output_tex
        };

        let filtered_tex = self.temporal(
            rg,
            &irradiance_tex,
            gbuffer_depth,
            reprojection_map,
            &reprojected_history_tex,
            &invalidity_output_tex,
            temporal_output_tex,
        );

        let filtered_tex = Self::spatial(
            rg,
            &filtered_tex,
            gbuffer_depth,
            ssao_tex,
            bindless_descriptor_set,
        );

        RtdgiOutput {
            screen_irradiance_tex: filtered_tex.into(),
            candidates: RtdgiCandidates {
                candidate_radiance_tex,
                candidate_normal_tex,
                candidate_hit_tex,
            },
        }
    }
}
