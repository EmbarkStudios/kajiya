use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration},
};
use kajiya_rg::{self as rg, SimpleRenderPass};
use rg::BindToSimpleRenderPass;

use super::surfel_gi::SurfelGiRenderState;

// Must match `wrc_settings.hlsl`
const WRC_GRID_DIMS: [usize; 3] = [8, 3, 8];
const WRC_PROBE_DIMS: usize = 32;
const WRC_ATLAS_PROBE_COUNT: [usize; 2] = [16, 16];

pub struct WrcRenderState {
    radiance_atlas: rg::Handle<Image>,
}

impl<'rg, RgPipelineHandle> BindToSimpleRenderPass<'rg, RgPipelineHandle> for WrcRenderState {
    fn bind(
        &self,
        pass: SimpleRenderPass<'rg, RgPipelineHandle>,
    ) -> SimpleRenderPass<'rg, RgPipelineHandle> {
        pass.read(&self.radiance_atlas)
    }
}

pub fn wrc_trace(
    rg: &mut rg::TemporalRenderGraph,
    surfel_gi: &SurfelGiRenderState,
    sky_cube: &rg::Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
    tlas: &rg::Handle<RayTracingAcceleration>,
) -> WrcRenderState {
    let total_probe_count: usize = WRC_GRID_DIMS.into_iter().product();
    let total_pixel_count = total_probe_count * WRC_PROBE_DIMS.pow(2);

    let mut radiance_atlas = rg.create(ImageDesc::new_2d(
        vk::Format::R32G32B32A32_SFLOAT,
        [
            (WRC_ATLAS_PROBE_COUNT[0] * WRC_PROBE_DIMS) as _,
            (WRC_ATLAS_PROBE_COUNT[1] * WRC_PROBE_DIMS) as _,
        ],
    ));

    SimpleRenderPass::new_rt(
        rg.add_pass("wrc trace"),
        "/shaders/wrc/trace_wrc.rgen.hlsl",
        &[
            "/shaders/rt/gbuffer.rmiss.hlsl",
            "/shaders/rt/shadow.rmiss.hlsl",
        ],
        &["/shaders/rt/gbuffer.rchit.hlsl"],
    )
    .read(sky_cube)
    .bind(surfel_gi)
    .write(&mut radiance_atlas)
    .raw_descriptor_set(1, bindless_descriptor_set)
    .trace_rays(tlas, [total_pixel_count as _, 1, 1]);

    WrcRenderState { radiance_atlas }
}

impl WrcRenderState {
    pub fn see_through(
        &self,
        rg: &mut rg::TemporalRenderGraph,
        sky_cube: &rg::Handle<Image>,
        surfel_gi: &SurfelGiRenderState,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
        output_img: &mut rg::Handle<Image>,
    ) {
        SimpleRenderPass::new_rt(
            rg.add_pass("wrc see through"),
            "/shaders/wrc/wrc_see_through.rgen.hlsl",
            &[
                "/shaders/rt/gbuffer.rmiss.hlsl",
                "/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/shaders/rt/gbuffer.rchit.hlsl"],
        )
        .bind(self)
        .read(sky_cube)
        .bind(surfel_gi)
        .write(output_img)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, output_img.desc().extent);
    }
}
