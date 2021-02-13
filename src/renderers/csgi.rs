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
                ImageDesc::new_3d(
                    vk::Format::R32G32B32A32_SFLOAT,
                    [32 * SLICE_COUNT as u32, 32, 32],
                )
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
        .constants(SLICE_DIRS)
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
        .constants((out_img.desc().extent_inv_extent_2d(), SLICE_DIRS))
        .dispatch(out_img.desc().extent);
    }
}

const SLICE_COUNT: usize = 16;
const SLICE_DIRS: [[f32; 4]; SLICE_COUNT] = [
    [0.85225946, 0.36958295, 0.37021905, 0.0],
    [0.263413, 0.9570777, 0.120897286, 0.0],
    [0.5356124, -0.75899905, -0.3701891, 0.0],
    [-0.97195244, 0.23506321, 0.0073164566, 0.0],
    [-0.544322, 0.38344887, 0.74611014, 0.0],
    [-0.6345936, -0.62937766, -0.448525, 0.0],
    [0.14410779, -0.35573938, -0.9234079, 0.0],
    [0.22534491, 0.5385581, -0.811896, 0.0],
    [-0.09385556, -0.98527056, 0.14294423, 0.0],
    [-0.7481385, -0.46306336, 0.47524846, 0.0],
    [0.08077474, -0.47626922, 0.8755816, 0.0],
    [-0.448539, 0.87202656, -0.1959147, 0.0],
    [0.8082192, -0.47252876, 0.3514234, 0.0],
    [0.22675002, 0.38531533, 0.8944922, 0.0],
    [0.87311435, 0.2334949, -0.4279624, 0.0],
    [-0.5878232, 0.17476612, -0.7898867, 0.0],
];
