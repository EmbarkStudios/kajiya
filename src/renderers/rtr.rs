use rg::GetOrCreateTemporal;
use slingshot::{
    ash::vk,
    backend::{image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
};

pub struct RtrRenderer {
    filtered_output_tex: rg::TemporalResourceKey,
    history_tex: rg::TemporalResourceKey,
}

impl Default for RtrRenderer {
    fn default() -> Self {
        Self {
            filtered_output_tex: "rtr.0".into(),
            history_tex: "rtr.1".into(),
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
        gbuffer: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
        reprojection_map: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> rg::Handle<Image> {
        let mut refl0_tex = rg.create(
            gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        let mut refl1_tex = rg.create(
            gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R32G32B32A32_SFLOAT),
        );

        SimpleRenderPass::new_rt(
            rg.add_pass(),
            "/assets/shaders/rtr/reflection.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .read(gbuffer)
        .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
        .write(&mut refl0_tex)
        .write(&mut refl1_tex)
        .constants(gbuffer.desc().extent_inv_extent_2d())
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, refl0_tex.desc().extent);

        let mut resolved_tex = rg.create(
            gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );

        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/rtr/resolve.hlsl")
            .read(gbuffer)
            .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
            .read(&refl0_tex)
            .read(&refl1_tex)
            .write(&mut resolved_tex)
            .constants(resolved_tex.desc().extent_inv_extent_2d())
            .dispatch(resolved_tex.desc().extent);

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

        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/rtr/temporal_filter.hlsl")
            .read(&resolved_tex)
            .read(&history_tex)
            .read(reprojection_map)
            .write(&mut filtered_output_tex)
            .constants(filtered_output_tex.desc().extent_inv_extent_2d())
            .dispatch(resolved_tex.desc().extent);

        filtered_output_tex
    }
}
