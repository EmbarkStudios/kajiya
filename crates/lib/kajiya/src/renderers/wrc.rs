use kajiya_backend::{
    ash::vk,
    vulkan::{image::*, ray_tracing::RayTracingAcceleration, shader::ShaderSource},
};
use kajiya_rg::{self as rg, SimpleRenderPass};
use rg::BindToSimpleRenderPass;

use super::ircache::IrcacheRenderState;

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
    ircache: &mut IrcacheRenderState,
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
        ShaderSource::hlsl("/shaders/wrc/trace_wrc.rgen.hlsl"),
        [
            ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
            ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
        ],
        [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
    )
    .read(sky_cube)
    .bind_mut(ircache)
    .write(&mut radiance_atlas)
    .raw_descriptor_set(1, bindless_descriptor_set)
    .trace_rays(tlas, [total_pixel_count as _, 1, 1]);

    WrcRenderState { radiance_atlas }
}

pub fn allocate_dummy_output(rg: &mut rg::TemporalRenderGraph) -> WrcRenderState {
    WrcRenderState {
        radiance_atlas: rg.create(ImageDesc::new_2d(vk::Format::R8_UNORM, [1, 1])),
    }
}

impl WrcRenderState {
    pub fn see_through(
        &self,
        rg: &mut rg::TemporalRenderGraph,
        sky_cube: &rg::Handle<Image>,
        ircache: &mut IrcacheRenderState,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
        output_img: &mut rg::Handle<Image>,
    ) {
        SimpleRenderPass::new_rt(
            rg.add_pass("wrc see through"),
            ShaderSource::hlsl("/shaders/wrc/wrc_see_through.rgen.hlsl"),
            [
                ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
            ],
            [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
        )
        .bind(self)
        .read(sky_cube)
        .bind_mut(ircache)
        .write(output_img)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(tlas, output_img.desc().extent);
    }
}
