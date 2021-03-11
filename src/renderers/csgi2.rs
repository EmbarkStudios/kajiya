// Cone sweep global illumination prototype

use std::sync::Arc;

use glam::Vec3;
use rg::GetOrCreateTemporal;
use slingshot::{
    ash::{version::DeviceV1_0, vk},
    backend::{
        image::*,
        ray_tracing::RayTracingAcceleration,
        shader::{
            PipelineShader, PipelineShaderDesc, RasterPipelineDesc, RenderPass, ShaderPipelineStage,
        },
    },
    rg::{self, BindRgRef, IntoRenderPassPipelineBinding, SimpleRenderPass},
    vk_sync::AccessType,
};

const VOLUME_DIMS: u32 = 64;

use super::GbufferDepth;

pub struct Csgi2Renderer {
    pub trace_subdiv: i32,
    pub neighbors_per_frame: i32,
}

impl Default for Csgi2Renderer {
    fn default() -> Self {
        Self {
            trace_subdiv: 3,
            neighbors_per_frame: 2,
        }
    }
}

pub struct Csgi2Volume {
    pub direct_cascade0: rg::Handle<Image>,
    pub indirect_cascade0: rg::Handle<Image>,
}

impl Csgi2Renderer {
    pub fn render(
        &mut self,
        eye_position: Vec3,
        rg: &mut rg::TemporalRenderGraph,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> Csgi2Volume {
        let mut direct_cascade0 = rg
            .get_or_create_temporal(
                "csgi2.direct_cascade0",
                ImageDesc::new_3d(
                    //vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::Format::R16G16B16A16_SFLOAT,
                    [
                        VOLUME_DIMS * PRETRACE_COUNT as u32,
                        VOLUME_DIMS,
                        VOLUME_DIMS,
                    ],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        let mut indirect_cascade0 = rg
            .get_or_create_temporal(
                "csgi2.indirect_cascade0",
                ImageDesc::new_3d(
                    //vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::Format::R16G16B16A16_SFLOAT,
                    [VOLUME_DIMS * TRACE_COUNT as u32, VOLUME_DIMS, VOLUME_DIMS],
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap();

        /*SimpleRenderPass::new_compute(
            rg.add_pass("csgi2 clear"),
            "/assets/shaders/csgi2/clear_volume.hlsl",
        )
        .write(&mut direct_cascade0)
        .dispatch(direct_cascade0.desc().extent);*/

        let sweep_vx_count = VOLUME_DIMS >> self.trace_subdiv.clamp(0, 5);

        SimpleRenderPass::new_rt(
            rg.add_pass("csgi2 trace"),
            "/assets/shaders/csgi2/trace_volume.rgen.hlsl",
            &[
                "/assets/shaders/rt/triangle.rmiss.hlsl",
                "/assets/shaders/rt/shadow.rmiss.hlsl",
            ],
            &["/assets/shaders/rt/triangle.rchit.hlsl"],
        )
        .write(&mut direct_cascade0)
        .constants(sweep_vx_count)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .trace_rays(
            tlas,
            [
                VOLUME_DIMS * PRETRACE_COUNT as u32,
                VOLUME_DIMS as u32,
                VOLUME_DIMS / sweep_vx_count,
            ],
        );

        SimpleRenderPass::new_compute(
            rg.add_pass("csgi2 sweep"),
            "/assets/shaders/csgi2/sweep_volume.hlsl",
        )
        .read(&direct_cascade0)
        .write(&mut indirect_cascade0)
        .dispatch([VOLUME_DIMS, VOLUME_DIMS, 1]);

        Csgi2Volume {
            direct_cascade0,
            indirect_cascade0,
        }
    }
}

impl Csgi2Volume {
    pub fn debug_raster_voxel_grid(
        &self,
        rg: &mut rg::RenderGraph,
        render_pass: Arc<RenderPass>,
        depth_img: &mut rg::Handle<Image>,
        color_img: &mut rg::Handle<Image>,
    ) {
        let mut pass = rg.add_pass("raster csgi2 voxels");

        let pipeline = pass.register_raster_pipeline(
            &[
                PipelineShader {
                    code: "/assets/shaders/csgi2/raster_voxels_vs.hlsl",
                    desc: PipelineShaderDesc::builder(ShaderPipelineStage::Vertex)
                        .build()
                        .unwrap(),
                },
                PipelineShader {
                    code: "/assets/shaders/csgi2/raster_voxels_ps.hlsl",
                    desc: PipelineShaderDesc::builder(ShaderPipelineStage::Pixel)
                        .build()
                        .unwrap(),
                },
            ],
            RasterPipelineDesc::builder()
                .render_pass(render_pass.clone())
                .face_cull(true),
        );

        let depth_ref = pass.raster(depth_img, AccessType::DepthStencilAttachmentWrite);
        let color_ref = pass.raster(color_img, AccessType::ColorAttachmentWrite);
        let grid_ref = pass.read(
            &self.direct_cascade0,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        pass.render(move |api| {
            let [width, height, _] = color_ref.desc().extent;

            api.begin_render_pass(
                &*render_pass,
                [width, height],
                &[(color_ref, &ImageViewDesc::default())],
                Some((
                    depth_ref,
                    &ImageViewDesc::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                        .build()
                        .unwrap(),
                )),
            );

            api.set_default_view_and_scissor([width, height]);

            let _ = api.bind_raster_pipeline(
                pipeline
                    .into_binding()
                    .descriptor_set(0, &[grid_ref.bind()]),
            );

            unsafe {
                let raw_device = &api.device().raw;
                let cb = api.cb;

                raw_device.cmd_draw(
                    cb.raw,
                    // 6 verts (two triangles) per cube face
                    6 * PRETRACE_COUNT as u32 * VOLUME_DIMS * VOLUME_DIMS * VOLUME_DIMS,
                    1,
                    0,
                    0,
                );
            }

            api.end_render_pass();
        });
    }
}

const PRETRACE_COUNT: usize = 6;
const TRACE_COUNT: usize = 6;
