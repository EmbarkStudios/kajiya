use slingshot::{
    ash::vk,
    backend::image::*,
    rg::{self, GetOrCreateTemporal, SimpleRenderPass},
};

pub struct SsgiRenderer {
    filtered_output_tex: rg::TemporalResourceKey,
    history_tex: rg::TemporalResourceKey,
}

impl Default for SsgiRenderer {
    fn default() -> Self {
        Self {
            filtered_output_tex: "ssgi.0".into(),
            history_tex: "ssgi.1".into(),
        }
    }
}

impl SsgiRenderer {
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
        reprojection_map: &rg::Handle<Image>,
        prev_radiance: &rg::Handle<Image>,
    ) -> rg::ReadOnlyHandle<Image> {
        let half_view_normal_tex = Self::extract_half_res_gbuffer_view_normal_rgba8(rg, gbuffer);
        let half_depth_tex = Self::extract_half_res_depth(rg, depth);

        let mut raw_ssgi_tex = rg.create(
            gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );
        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/ssgi/ssgi.hlsl")
            .read(gbuffer)
            .read(&half_depth_tex)
            .read(&half_view_normal_tex)
            .read(prev_radiance)
            .read(reprojection_map)
            .write(&mut raw_ssgi_tex)
            .constants((
                gbuffer.desc().extent_inv_extent_2d(),
                raw_ssgi_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(raw_ssgi_tex.desc().extent);

        let mut ssgi_tex = rg.create(
            gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );
        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/ssgi/spatial_filter.hlsl")
            .read(&raw_ssgi_tex)
            .read(&half_depth_tex)
            .read(&half_view_normal_tex)
            .write(&mut ssgi_tex)
            .dispatch(ssgi_tex.desc().extent);

        let ssgi_tex = Self::upsample_ssgi(rg, &ssgi_tex, depth, gbuffer);

        let history_tex = rg
            .get_or_create_temporal(
                self.history_tex.clone(),
                Self::temporal_tex_desc(gbuffer.desc().extent_2d()),
            )
            .unwrap();

        let mut filtered_output_tex = rg
            .get_or_create_temporal(
                self.filtered_output_tex.clone(),
                Self::temporal_tex_desc(gbuffer.desc().extent_2d()),
            )
            .unwrap();

        std::mem::swap(&mut self.filtered_output_tex, &mut self.history_tex);

        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/ssgi/temporal_filter.hlsl")
            .read(&ssgi_tex)
            .read(&history_tex)
            .read(reprojection_map)
            .write(&mut filtered_output_tex)
            .constants(filtered_output_tex.desc().extent_inv_extent_2d())
            .dispatch(ssgi_tex.desc().extent);

        filtered_output_tex.into()
    }

    fn temporal_tex_desc(extent: [u32; 2]) -> ImageDesc {
        ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, extent)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
    }

    fn extract_half_res_gbuffer_view_normal_rgba8(
        rg: &mut rg::RenderGraph,
        gbuffer: &rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let mut output_tex = rg.create(
            gbuffer
                .desc()
                .half_res()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R8G8B8A8_SNORM),
        );
        SimpleRenderPass::new_compute(
            rg.add_pass(),
            "/assets/shaders/extract_half_res_gbuffer_view_normal_rgba8.hlsl",
        )
        .read(gbuffer)
        .write(&mut output_tex)
        .constants((
            gbuffer.desc().extent_inv_extent_2d(),
            output_tex.desc().extent_inv_extent_2d(),
        ))
        .dispatch(output_tex.desc().extent);
        output_tex
    }

    fn extract_half_res_depth(
        rg: &mut rg::RenderGraph,
        depth: &rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let mut output_tex = rg.create(
            depth
                .desc()
                .half_res()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R32_SFLOAT),
        );
        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/downscale_r.hlsl")
            .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
            .write(&mut output_tex)
            .constants((
                depth.desc().extent_inv_extent_2d(),
                output_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(output_tex.desc().extent);
        output_tex
    }

    fn upsample_ssgi(
        rg: &mut rg::RenderGraph,
        ssgi: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
        gbuffer: &rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let mut output_tex = rg.create(gbuffer.desc().format(vk::Format::R16G16B16A16_SFLOAT));

        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/ssgi/upsample.hlsl")
            .read(ssgi)
            .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
            .read(gbuffer)
            .write(&mut output_tex)
            .dispatch(output_tex.desc().extent);
        output_tex
    }
}
