use std::sync::Arc;

use kajiya_backend::{
    ash::{version::DeviceV1_0, vk},
    vk_sync::AccessType,
    vulkan::{buffer::*, image::*, shader::*},
};
use kajiya_rg::{self as rg};
use rg::{IntoRenderPassPipelineBinding, RenderGraph};

use crate::world_renderer::MeshInstance;

use super::GbufferDepth;

#[derive(Clone)]
pub struct UploadedTriMesh {
    pub index_buffer_offset: u64,
    pub index_count: u32,
}

pub struct RasterMeshesData<'a> {
    pub meshes: &'a [UploadedTriMesh],
    pub instances: &'a [MeshInstance],
    pub vertex_buffer: Arc<Buffer>,
    pub bindless_descriptor_set: vk::DescriptorSet,
}

pub fn raster_meshes(
    rg: &mut RenderGraph,
    render_pass: Arc<RenderPass>,
    gbuffer_depth: &mut GbufferDepth,
    velocity_img: &mut rg::Handle<Image>,
    mesh_data: RasterMeshesData<'_>,
) {
    let mut pass = rg.add_pass("raster simple");

    let pipeline = pass.register_raster_pipeline(
        &[
            PipelineShader {
                code: "/assets/shaders/raster_simple_vs.hlsl",
                desc: PipelineShaderDesc::builder(ShaderPipelineStage::Vertex)
                    .build()
                    .unwrap(),
            },
            PipelineShader {
                code: "/assets/shaders/raster_simple_ps.hlsl",
                desc: PipelineShaderDesc::builder(ShaderPipelineStage::Pixel)
                    .build()
                    .unwrap(),
            },
        ],
        RasterPipelineDesc::builder()
            .render_pass(render_pass.clone())
            .face_cull(true)
            .push_constants_bytes(7 * std::mem::size_of::<u32>()),
    );

    let meshes: Vec<UploadedTriMesh> = mesh_data.meshes.to_vec();
    let instances: Vec<MeshInstance> = mesh_data.instances.to_vec();

    let depth_ref = pass.raster(
        &mut gbuffer_depth.depth,
        AccessType::DepthStencilAttachmentWrite,
    );

    let geometric_normal_ref = pass.raster(
        &mut gbuffer_depth.geometric_normal,
        AccessType::ColorAttachmentWrite,
    );
    let gbuffer_ref = pass.raster(&mut gbuffer_depth.gbuffer, AccessType::ColorAttachmentWrite);
    let velocity_ref = pass.raster(velocity_img, AccessType::ColorAttachmentWrite);

    let vertex_buffer = mesh_data.vertex_buffer.clone();
    let bindless_descriptor_set = mesh_data.bindless_descriptor_set;

    pass.render(move |api| {
        let [width, height, _] = gbuffer_ref.desc().extent;

        api.begin_render_pass(
            &*render_pass,
            [width, height],
            &[
                (geometric_normal_ref, &ImageViewDesc::default()),
                (gbuffer_ref, &ImageViewDesc::default()),
                (velocity_ref, &ImageViewDesc::default()),
            ],
            Some((
                depth_ref,
                &ImageViewDesc::builder()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    .build()
                    .unwrap(),
            )),
        );

        api.set_default_view_and_scissor([width, height]);

        let pipeline = api.bind_raster_pipeline(
            pipeline
                .into_binding()
                .descriptor_set(0, &[])
                .raw_descriptor_set(1, bindless_descriptor_set),
        );

        unsafe {
            let raw_device = &api.device().raw;
            let cb = api.cb;

            for instance in instances {
                let mesh = &meshes[instance.mesh.0];

                raw_device.cmd_bind_index_buffer(
                    cb.raw,
                    vertex_buffer.raw,
                    mesh.index_buffer_offset,
                    vk::IndexType::UINT32,
                );

                let push_constants = (
                    instance.mesh.0 as u32,
                    instance.position.x,
                    instance.position.y,
                    instance.position.z,
                    instance.prev_position.x,
                    instance.prev_position.y,
                    instance.prev_position.z,
                );

                pipeline.push_constants(
                    cb.raw,
                    vk::ShaderStageFlags::ALL_GRAPHICS,
                    0,
                    std::slice::from_raw_parts(
                        &push_constants as *const _ as *const u8,
                        std::mem::size_of_val(&push_constants),
                    ),
                );

                raw_device.cmd_draw_indexed(cb.raw, mesh.index_count, 1, 0, 0, 0);
            }
        }

        api.end_render_pass();
    });
}
