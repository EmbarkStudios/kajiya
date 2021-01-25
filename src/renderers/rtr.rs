use std::{mem::size_of, sync::Arc};

use slingshot::{
    ash::vk,
    backend::{
        buffer::{Buffer, BufferDesc},
        device,
        image::*,
        ray_tracing::RayTracingAcceleration,
        shader::*,
    },
    rg::{self, BindRgRef, SimpleRenderPass},
    vk_sync::AccessType,
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

        SimpleRenderPass::new_rt(
            rg.add_pass(),
            "/assets/shaders/rt/reflection.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .read(gbuffer)
        .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
        .write(&mut refl0_tex)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, refl0_tex.desc().extent);

        refl0_tex
    }
}
