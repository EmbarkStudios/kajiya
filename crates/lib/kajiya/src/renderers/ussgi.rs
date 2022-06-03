use super::{GbufferDepth, PingPongTemporalResource};
use kajiya_backend::{
    ash::vk::{self, ImageAspectFlags},
    vulkan::image::*,
};
use kajiya_rg::{self as rg, SimpleRenderPass};

pub struct UssgiRenderer {
    ussgi_tex: PingPongTemporalResource,
}

impl Default for UssgiRenderer {
    fn default() -> Self {
        Self {
            ussgi_tex: PingPongTemporalResource::new("ussgi"),
        }
    }
}

const TEX_FMT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

impl UssgiRenderer {
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        reprojection_map: &rg::Handle<Image>,
        prev_radiance: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
    ) -> rg::ReadOnlyHandle<Image> {
        let gbuffer_desc = gbuffer_depth.gbuffer.desc();
        let half_view_normal_tex = gbuffer_depth.half_view_normal(rg);

        let mut ussgi_tex = rg.create(
            gbuffer_desc
                .usage(vk::ImageUsageFlags::empty())
                .format(TEX_FMT),
        );

        SimpleRenderPass::new_compute(rg.add_pass("ussgi"), "/shaders/ssgi/ussgi.hlsl")
            .read(&gbuffer_depth.gbuffer)
            .read_aspect(&gbuffer_depth.depth, ImageAspectFlags::DEPTH)
            .read(&*half_view_normal_tex)
            .read(prev_radiance)
            .read(reprojection_map)
            .write(&mut ussgi_tex)
            .raw_descriptor_set(1, bindless_descriptor_set)
            .constants((
                gbuffer_desc.extent_inv_extent_2d(),
                ussgi_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(ussgi_tex.desc().extent);

        let (mut filtered_output_tex, history_tex) = self
            .ussgi_tex
            .get_output_and_history(rg, Self::temporal_tex_desc(gbuffer_desc.extent_2d()));

        SimpleRenderPass::new_compute(
            rg.add_pass("ussgi temporal"),
            "/shaders/ssgi/temporal_filter.hlsl",
        )
        .read(&ussgi_tex)
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
}
