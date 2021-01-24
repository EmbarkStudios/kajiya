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
    rg::{self, BindRgRef, SimpleComputePass},
    vk_sync::AccessType,
};

use crate::temporal::*;

pub struct SsgiRenderer {
    pub temporal0: Temporal<Image>,
    pub temporal1: Temporal<Image>,
}

pub struct SsgiRenderInstance {
    pub temporal0: rg::Handle<Image>,
    pub temporal1: rg::Handle<Image>,
}

impl SsgiRenderer {
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
        SsgiRenderer {
            temporal0: Self::make_temporal_tex(device, extent),
            temporal1: Self::make_temporal_tex(device, extent),
        }
    }

    fn on_begin(&mut self) {
        std::mem::swap(&mut self.temporal0, &mut self.temporal1);
    }

    crate::impl_renderer_temporal_logic! {
        SsgiRenderInstance,
        temporal0,
        temporal1,
    }
}

impl SsgiRenderInstance {
    fn extract_half_res_gbuffer_view_normal_rgba8(
        rg: &mut rg::RenderGraph,
        gbuffer: &rg::Handle<Image>,
    ) -> rg::Handle<Image> {
        let mut pass = rg.add_pass();
        let mut output_tex = pass.create(
            &gbuffer
                .desc()
                .half_res()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R8G8B8A8_SNORM),
        );
        SimpleComputePass::new(
            pass,
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
        let mut pass = rg.add_pass();
        let mut output_tex = pass.create(
            &depth
                .desc()
                .half_res()
                .usage(vk::ImageUsageFlags::empty())
                .format(vk::Format::R32_SFLOAT),
        );
        SimpleComputePass::new(pass, "/assets/shaders/downscale_r.hlsl")
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
        let mut pass = rg.add_pass();
        let mut output_tex = pass.create(&gbuffer.desc().format(vk::Format::R16G16B16A16_SFLOAT));

        SimpleComputePass::new(pass, "/assets/shaders/ssgi/upsample.hlsl")
            .read(ssgi)
            .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
            .read(gbuffer)
            .write(&mut output_tex)
            .dispatch(output_tex.desc().extent);
        output_tex
    }

    pub fn render(
        &mut self,
        rg: &mut rg::RenderGraph,
        gbuffer: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
        reprojection_map: &rg::Handle<Image>,
    ) -> &rg::Handle<Image> {
        let half_view_normal_tex = Self::extract_half_res_gbuffer_view_normal_rgba8(rg, gbuffer);
        let half_depth_tex = Self::extract_half_res_depth(rg, depth);

        let mut pass = rg.add_pass();
        let mut raw_ssgi_tex = pass.create(
            &gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );
        SimpleComputePass::new(pass, "/assets/shaders/ssgi/ssgi.hlsl")
            .read(gbuffer)
            .read(&half_depth_tex)
            .read(&half_view_normal_tex)
            .write(&mut raw_ssgi_tex)
            .constants((
                gbuffer.desc().extent_inv_extent_2d(),
                raw_ssgi_tex.desc().extent_inv_extent_2d(),
            ))
            .dispatch(raw_ssgi_tex.desc().extent);

        let mut pass = rg.add_pass();
        let mut ssgi_tex = pass.create(
            &gbuffer
                .desc()
                .usage(vk::ImageUsageFlags::empty())
                .half_res()
                .format(vk::Format::R16G16B16A16_SFLOAT),
        );
        SimpleComputePass::new(pass, "/assets/shaders/ssgi/spatial_filter.hlsl")
            .read(&raw_ssgi_tex)
            .read(&half_depth_tex)
            .read(&half_view_normal_tex)
            .write(&mut ssgi_tex)
            .dispatch(ssgi_tex.desc().extent);

        let ssgi_tex = Self::upsample_ssgi(rg, &ssgi_tex, depth, gbuffer);

        let filtered_output_tex = &mut self.temporal0;
        let history_tex = &self.temporal1;

        SimpleComputePass::new(rg.add_pass(), "/assets/shaders/ssgi/temporal_filter.hlsl")
            .read(&ssgi_tex)
            .read(history_tex)
            .read(reprojection_map)
            .write(filtered_output_tex)
            .constants(filtered_output_tex.desc().extent_inv_extent_2d())
            .dispatch(ssgi_tex.desc().extent);

        filtered_output_tex
    }
}
