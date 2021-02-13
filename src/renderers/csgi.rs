// Cone sweep global illumination prototype

use rg::GetOrCreateTemporal;
use slingshot::{
    ash::vk,
    backend::{buffer::*, image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
    vk_sync, Device,
};

use super::GbufferDepth;

pub struct CsgiRenderer;

pub struct CsgiVolume {
    pub dir0: rg::Handle<Image>,
}

impl CsgiRenderer {
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> CsgiVolume {
        let mut dir0 = rg
            .get_or_create_temporal(
                "csgi.dir0",
                ImageDesc::new_3d(vk::Format::R32G32B32A32_SFLOAT, [32, 32, 32])
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        /*SimpleRenderPass::new_compute(
            rg.add_pass("csgi clear"),
            "/assets/shaders/csgi/clear_volume.hlsl",
        )
        .write(&mut dir0)
        .dispatch(dir0.desc().extent);*/

        SimpleRenderPass::new_rt(
            rg.add_pass("csgi trace"),
            "/assets/shaders/csgi/trace_volume.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .write(&mut dir0)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, dir0.desc().extent);

        CsgiVolume { dir0 }
    }
}

impl CsgiVolume {
    pub fn render_debug(
        &self,
        rg: &mut rg::TemporalRenderGraph,
        gbuffer_depth: &GbufferDepth,
        out_img: &mut rg::Handle<Image>,
    ) {
        SimpleRenderPass::new_compute(
            rg.add_pass("csgi debug"),
            "/assets/shaders/csgi/render_debug.hlsl",
        )
        .read(&gbuffer_depth.gbuffer)
        .read_aspect(&gbuffer_depth.depth, vk::ImageAspectFlags::DEPTH)
        .read(&self.dir0)
        .write(out_img)
        .constants(out_img.desc().extent_inv_extent_2d())
        .dispatch(out_img.desc().extent);
    }
}
