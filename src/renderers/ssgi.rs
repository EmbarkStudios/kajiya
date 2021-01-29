use slingshot::{
    ash::vk,
    backend::image::*,
    rg::{self, GetOrCreateTemporal, SimpleRenderPass},
};

use super::{
    half_res::{extract_half_res_depth, extract_half_res_gbuffer_view_normal_rgba8},
    GbufferDepth,
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
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        prev_radiance: &rg::Handle<Image>,
    ) -> rg::ReadOnlyHandle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();
        let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);
        let half_depth_tex = gbuffer_depth.half_depth(rg);

        let mut raw_ssgi_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );
        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/ssgi/ssgi.hlsl")
            .read(&gbuffer_depth.gbuffer)
            .read(&*half_depth_tex)
            .read(&*half_view_normal_tex)
            .read(prev_radiance)
            .read(reprojection_map)
            .write(&mut raw_ssgi_tex)
            .constants((
                gbuffer_desc.extent_inv_extent_2d(),
                raw_ssgi_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(raw_ssgi_tex.desc().extent);

        let mut ssgi_tex = rg.create(
            gbuffer_desc
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

        let ssgi_tex =
            Self::upsample_ssgi(rg, &ssgi_tex, &gbuffer_depth.depth, &gbuffer_depth.gbuffer);

        let history_tex = rg
            .get_or_create_temporal(
                self.history_tex.clone(),
                Self::temporal_tex_desc(gbuffer_desc.extent_2d()),
            )
            .unwrap();

        let mut filtered_output_tex = rg
            .get_or_create_temporal(
                self.filtered_output_tex.clone(),
                Self::temporal_tex_desc(gbuffer_desc.extent_2d()),
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
