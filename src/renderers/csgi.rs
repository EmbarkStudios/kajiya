// Cone sweep global illumination prototype

use rg::GetOrCreateTemporal;
use slingshot::{
    ash::vk,
    backend::{image::*, ray_tracing::RayTracingAcceleration},
    rg::{self, SimpleRenderPass},
};

const VOLUME_DIMS: u32 = 32;

use super::GbufferDepth;

pub struct CsgiRenderer {
    pub trace_subdiv: i32,
    pub neighbors_per_frame: i32,
}

impl Default for CsgiRenderer {
    fn default() -> Self {
        Self {
            trace_subdiv: 3,
            neighbors_per_frame: 2,
        }
    }
}

pub struct CsgiVolume {
    pub cascade0: rg::Handle<Image>,
}

impl CsgiRenderer {
    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> CsgiVolume {
        let mut cascade0 = rg
            .get_or_create_temporal(
                "csgi.cascade0",
                ImageDesc::new_3d(
                    vk::Format::B10G11R11_UFLOAT_PACK32,
                    [VOLUME_DIMS * SLICE_COUNT as u32, VOLUME_DIMS, VOLUME_DIMS],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        let mut cascade0_integr = rg
            .get_or_create_temporal(
                "csgi.cascade0_integr",
                ImageDesc::new_3d(
                    vk::Format::R16G16B16A16_SFLOAT,
                    [
                        VOLUME_DIMS * SLICE_COUNT as u32,
                        9 * VOLUME_DIMS,
                        VOLUME_DIMS,
                    ],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        /*SimpleRenderPass::new_compute(
            rg.add_pass("csgi clear"),
            "/assets/shaders/csgi/clear_volume.hlsl",
        )
        .write(&mut cascade0)
        .dispatch(cascade0.desc().extent);*/

        let sweep_vx_count = VOLUME_DIMS >> self.trace_subdiv.clamp(0, 5);
        let neighbors_per_frame = self.neighbors_per_frame.clamp(1, 9);

        SimpleRenderPass::new_rt(
            rg.add_pass("csgi trace"),
            "/assets/shaders/csgi/trace_volume.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .read(&mut cascade0)
        .write(&mut cascade0_integr)
        .constants((SLICE_DIRS, (sweep_vx_count, neighbors_per_frame)))
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(
            tlas,
            [
                VOLUME_DIMS * SLICE_COUNT as u32,
                VOLUME_DIMS * neighbors_per_frame as u32,
                VOLUME_DIMS / sweep_vx_count,
            ],
        );

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi sweep"),
            "/assets/shaders/csgi/sweep_volume.hlsl",
        )
        .read(&cascade0_integr)
        .write(&mut cascade0)
        .dispatch(cascade0.desc().extent);

        /*let mut pretrace_hit_img = rg.create(
                    ImageDesc::new_3d(
                        vk::Format::R8_UNORM,
                        [
                            PRETRACE_DIMS * PRETRACE_COUNT as u32,
                            PRETRACE_DIMS,
                            PRETRACE_DIMS,
                        ],
                    )
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                );

                let mut pretrace_normal_img = rg.create(
                    ImageDesc::new_3d(
                        vk::Format::R8G8B8A8_UNORM,
                        [
                            PRETRACE_DIMS * PRETRACE_COUNT as u32,
                            PRETRACE_DIMS,
                            PRETRACE_DIMS,
                        ],
                    )
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                );

                let mut pretrace_col_img = rg.create(
                    ImageDesc::new_3d(
                        vk::Format::R16G16B16A16_SFLOAT,
                        [
                            PRETRACE_DIMS * PRETRACE_COUNT as u32,
                            PRETRACE_DIMS,
                            PRETRACE_DIMS,
                        ],
                    )
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                );

                SimpleRenderPass::new_compute(
                    rg.add_pass("csgi clear"),
                    "/assets/shaders/csgi/clear_volume.hlsl",
                )
                .write(&mut pretrace_hit_img)
                //.dispatch(pretrace_img.desc().div_up_extent([2, 2, 2]).extent);
                .dispatch(pretrace_col_img.desc().extent);
        */

        CsgiVolume {
            cascade0,
            //alt_cascade0,
        }
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
        .read(&self.cascade0)
        //.read(&self.alt_cascade0)
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
