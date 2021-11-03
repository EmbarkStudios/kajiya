use super::{GbufferDepth, PingPongTemporalResource};
use kajiya_backend::{ash::vk, vulkan::image::*};
use kajiya_rg::{self as rg, SimpleRenderPass, TemporalRenderGraph};

pub struct SsgiRenderer {
    ssgi_tex: PingPongTemporalResource,
}

impl Default for SsgiRenderer {
    fn default() -> Self {
        Self {
            ssgi_tex: PingPongTemporalResource::new("ssgi"),
        }
    }
}

const TEX_FMT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

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

        let mut ssgi_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(TEX_FMT),
        );

        SimpleRenderPass::new_compute(rg.add_pass("ssgi"), "/shaders/ssgi/ssgi.hlsl")
            .read(&gbuffer_depth.gbuffer)
            .read(&*half_depth_tex)
            .read(&*half_view_normal_tex)
            .read(prev_radiance)
            .read(reprojection_map)
            .write(&mut ssgi_tex)
            .constants((
                gbuffer_desc.extent_inv_extent_2d(),
                ssgi_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(ssgi_tex.desc().extent);

        Self::filter_ssgi(
            rg,
            &ssgi_tex,
            gbuffer_depth,
            reprojection_map,
            &mut self.ssgi_tex,
        )
    }

    fn filter_ssgi(
        rg: &mut TemporalRenderGraph,
        input: &rg::Handle<Image>,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        temporal_tex: &mut PingPongTemporalResource,
    ) -> rg::ReadOnlyHandle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();
        let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);
        let half_depth_tex = gbuffer_depth.half_depth(rg);

        let upsampled_tex = {
            let mut spatially_filtered_tex = rg.create(
                gbuffer_desc
                    .usage(vk::ImageUsageFlags::empty())
                    .half_res()
                    .format(TEX_FMT),
            );

            SimpleRenderPass::new_compute(
                rg.add_pass("ssgi spatial"),
                "/shaders/ssgi/spatial_filter.hlsl",
            )
            .read(input)
            .read(&half_depth_tex)
            .read(&half_view_normal_tex)
            .write(&mut spatially_filtered_tex)
            .dispatch(spatially_filtered_tex.desc().extent);

            Self::upsample_ssgi(
                rg,
                &spatially_filtered_tex,
                &gbuffer_depth.depth,
                &gbuffer_depth.gbuffer,
            )
        };

        let (mut filtered_output_tex, history_tex) = temporal_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(gbuffer_desc.extent_2d()));

        SimpleRenderPass::new_compute(
            rg.add_pass("ssgi temporal"),
            "/shaders/ssgi/temporal_filter.hlsl",
        )
        .read(&upsampled_tex)
        .read(&history_tex)
        .read(reprojection_map)
        .write(&mut filtered_output_tex)
        .constants(filtered_output_tex.desc().extent_inv_extent_2d())
        .dispatch(filtered_output_tex.desc().extent);

        filtered_output_tex.into()
    }

    fn temporal_tex_desc(extent: [u32; 2]) -> ImageDesc {
        ImageDesc::new_2d(TEX_FMT, extent)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
    }

    fn upsample_ssgi(
        rg: &mut rg::RenderGraph,
        ssgi: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
        gbuffer: &rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let mut output_tex = rg.create(gbuffer.desc().format(TEX_FMT));

        SimpleRenderPass::new_compute(rg.add_pass("ssgi upsample"), "/shaders/ssgi/upsample.hlsl")
            .read(ssgi)
            .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
            .read(gbuffer)
            .write(&mut output_tex)
            .dispatch(output_tex.desc().extent);
        output_tex
    }
}
