use std::sync::Arc;

use slingshot::{
    ash::vk,
    backend::{device, image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
};

use crate::temporal::*;

pub struct RtrRenderer {
    pub temporal0: Temporal<Image>,
    pub temporal1: Temporal<Image>,
}

pub struct RtrRenderInstance {
    pub temporal0: rg::Handle<Image>,
    pub temporal1: rg::Handle<Image>,
}

impl RtrRenderer {
    fn make_temporal_tex(device: &device::Device, extent: [u32; 2]) -> Temporal<Image> {
        Temporal::new(Arc::new(
            device
                .create_image(
                    ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, extent)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                    None,
                )
                .unwrap(),
        ))
    }

    pub fn new(device: &device::Device, extent: [u32; 2]) -> Self {
        RtrRenderer {
            temporal0: Self::make_temporal_tex(device, extent),
            temporal1: Self::make_temporal_tex(device, extent),
        }
    }

    fn on_begin(&mut self) {
        std::mem::swap(&mut self.temporal0, &mut self.temporal1);
    }

    crate::impl_renderer_temporal_logic! {
        RtrRenderInstance,
        temporal0,
        temporal1,
    }
}

impl RtrRenderInstance {
    pub fn render(
        &mut self,
        rg: &mut rg::RenderGraph,
        gbuffer: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
        reprojection_map: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> &rg::Handle<Image> {
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

        let filtered_output_tex = &mut self.temporal0;
        let history_tex = &self.temporal1;

        SimpleRenderPass::new_compute(rg.add_pass(), "/assets/shaders/rtr/temporal_filter.hlsl")
            .read(&resolved_tex)
            .read(history_tex)
            .read(reprojection_map)
            .write(filtered_output_tex)
            .constants(filtered_output_tex.desc().extent_inv_extent_2d())
            .dispatch(resolved_tex.desc().extent);

        filtered_output_tex
    }
}
