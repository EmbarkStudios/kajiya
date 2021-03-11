use std::sync::Arc;

use slingshot::{
    ash::{version::DeviceV1_0, vk},
    backend::ray_tracing::RayTracingAcceleration,
};

use crate::{
    backend::image::ImageViewDesc,
    backend::shader::*,
    renderers::{
        csgi::{self, CSGI_SLICE_DIRS},
        csgi2,
    },
    rg::*,
};

pub fn clear_depth(rg: &mut RenderGraph, img: &mut Handle<Image>) {
    let mut pass = rg.add_pass("clear depth");
    let output_ref = pass.write(img, AccessType::TransferWrite);

    pass.render(move |api| {
        let raw_device = &api.device().raw;
        let cb = api.cb;

        let image = api.resources.image(output_ref);

        unsafe {
            raw_device.cmd_clear_depth_stencil_image(
                cb.raw,
                image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearDepthStencilValue {
                    depth: 0f32,
                    stencil: 0,
                },
                std::slice::from_ref(&vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                }),
            );
        }
    });
}

pub fn clear_color(rg: &mut RenderGraph, img: &mut Handle<Image>, clear_color: [f32; 4]) {
    let mut pass = rg.add_pass("clear color");
    let output_ref = pass.write(img, AccessType::TransferWrite);

    pass.render(move |api| {
        let raw_device = &api.device().raw;
        let cb = api.cb;

        let image = api.resources.image(output_ref);

        unsafe {
            raw_device.cmd_clear_color_image(
                cb.raw,
                image.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: clear_color,
                },
                std::slice::from_ref(&vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                }),
            );
        }
    });
}

/*pub fn raymarch_sdf(
    rg: &mut RenderGraph,
    sdf_img: &Handle<Image>,
    desc: ImageDesc,
) -> Handle<Image> {
    let mut pass = rg.add_pass();

    let pipeline = pass.register_compute_pipeline("/assets/shaders/sdf/sdf_raymarch_gbuffer.hlsl");

    let sdf_ref = pass.read(
        sdf_img,
        AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
    );
    let mut output = pass.create(&desc);
    let output_ref = pass.write(&mut output, AccessType::ComputeShaderWrite);

    pass.render(move |api| {
        let pipeline = api.bind_compute_pipeline(pipeline.into_binding().descriptor_set(
            0,
            &[
                output_ref.bind(ImageViewDescBuilder::default()),
                sdf_ref.bind(ImageViewDescBuilder::default()),
            ],
        ));

        pipeline.dispatch(desc.extent);
    });

    output
}

pub fn edit_sdf(rg: &mut RenderGraph, sdf_img: &mut Handle<Image>, clear: bool) {
    let mut pass = rg.add_pass();

    let sdf_img_ref = pass.write(sdf_img, AccessType::ComputeShaderWrite);

    let pipeline_path = if clear {
        "/assets/shaders/sdf/gen_empty_sdf.hlsl"
    } else {
        "/assets/shaders/sdf/edit_sdf.hlsl"
    };

    let pipeline = pass.register_compute_pipeline(pipeline_path);

    pass.render(move |api| {
        let pipeline = api.bind_compute_pipeline(
            pipeline
                .into_binding()
                .descriptor_set(0, &[sdf_img_ref.bind(ImageViewDescBuilder::default())]),
        );
        pipeline.dispatch([SDF_DIM, SDF_DIM, SDF_DIM]);
    });
}

fn clear_sdf_bricks_meta(rg: &mut RenderGraph) -> Handle<Buffer> {
    let mut pass = rg.add_pass();

    let mut brick_meta_buf = pass.create(&BufferDesc {
        size: 20, // size of VkDrawIndexedIndirectCommand
        usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
    });
    let brick_meta_buf_ref = pass.write(&mut brick_meta_buf, AccessType::ComputeShaderWrite);

    let clear_meta_pipeline =
        pass.register_compute_pipeline("/assets/shaders/sdf/clear_bricks_meta.hlsl");

    pass.render(move |api| {
        let pipeline = api.bind_compute_pipeline(
            clear_meta_pipeline
                .into_binding()
                .descriptor_set(0, &[brick_meta_buf_ref.bind()]),
        );
        pipeline.dispatch([1, 1, 1]);
    });

    brick_meta_buf
}

pub struct SdfRasterBricks {
    pub brick_meta_buffer: Handle<Buffer>,
    pub brick_inst_buffer: Handle<Buffer>,
}

pub fn calculate_sdf_bricks_meta(rg: &mut RenderGraph, sdf_img: &Handle<Image>) -> SdfRasterBricks {
    let mut brick_meta_buf = clear_sdf_bricks_meta(rg);

    let mut pass = rg.add_pass();

    let sdf_ref = pass.read(
        sdf_img,
        AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
    );

    let brick_meta_buf_ref = pass.write(&mut brick_meta_buf, AccessType::ComputeShaderWrite);

    let mut brick_inst_buf = pass.create(&BufferDesc {
        size: (SDF_DIM as usize).pow(3) * 4 * 4,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
    });
    let brick_inst_buf_ref = pass.write(&mut brick_inst_buf, AccessType::ComputeShaderWrite);

    let calc_meta_pipeline = pass.register_compute_pipeline("/assets/shaders/sdf/find_bricks.hlsl");

    pass.render(move |api| {
        let pipeline = api.bind_compute_pipeline(calc_meta_pipeline.into_binding().descriptor_set(
            0,
            &[
                sdf_ref.bind(ImageViewDescBuilder::default()),
                brick_meta_buf_ref.bind(),
                brick_inst_buf_ref.bind(),
            ],
        ));
        pipeline.dispatch([SDF_DIM / 2, SDF_DIM / 2, SDF_DIM / 2]);
    });

    SdfRasterBricks {
        brick_meta_buffer: brick_meta_buf,
        brick_inst_buffer: brick_inst_buf,
    }
}

pub struct RasterSdfData<'a> {
    pub sdf_img: &'a Handle<Image>,
    pub brick_inst_buffer: &'a Handle<Buffer>,
    pub brick_meta_buffer: &'a Handle<Buffer>,
    pub cube_index_buffer: &'a Handle<Buffer>,
}

pub fn raster_sdf(
    rg: &mut RenderGraph,
    render_pass: Arc<RenderPass>,
    depth_img: &mut Handle<Image>,
    color_img: &mut Handle<Image>,
    raster_sdf_data: RasterSdfData<'_>,
) {
    let mut pass = rg.add_pass();

    let pipeline = pass.register_raster_pipeline(
        &[
            RasterPipelineShader {
                code: "/assets/shaders/raster_simple_vs.hlsl",
                desc: RasterShaderDesc::builder(RasterStage::Vertex)
                    .build()
                    .unwrap(),
            },
            RasterPipelineShader {
                code: "/assets/shaders/raster_simple_ps.hlsl",
                desc: RasterShaderDesc::builder(RasterStage::Pixel)
                    .build()
                    .unwrap(),
            },
        ],
        RasterPipelineDesc::builder()
            .render_pass(render_pass.clone())
            .face_cull(true),
    );

    let sdf_ref = pass.read(
        raster_sdf_data.sdf_img,
        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
    );
    let brick_inst_buffer = pass.read(
        raster_sdf_data.brick_inst_buffer,
        AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer,
    );
    let brick_meta_buffer = pass.read(
        raster_sdf_data.brick_meta_buffer,
        AccessType::IndirectBuffer,
    );
    let cube_index_buffer = pass.read(raster_sdf_data.cube_index_buffer, AccessType::IndexBuffer);

    let depth_ref = pass.raster(depth_img, AccessType::DepthStencilAttachmentWrite);
    let color_ref = pass.raster(color_img, AccessType::ColorAttachmentWrite);

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

        let _pipeline = api.bind_raster_pipeline(pipeline.into_binding().descriptor_set(
            0,
            &[
                brick_inst_buffer.bind(),
                sdf_ref.bind(ImageViewDescBuilder::default()),
            ],
        ));

        unsafe {
            let raw_device = &api.device().raw;
            let cb = api.cb;

            raw_device.cmd_bind_index_buffer(
                cb.raw,
                api.resources.buffer(cube_index_buffer).raw,
                0,
                vk::IndexType::UINT32,
            );

            raw_device.cmd_draw_indexed_indirect(
                cb.raw,
                api.resources.buffer(brick_meta_buffer).raw,
                0,
                1,
                0,
            );
        }

        api.end_render_pass();
    });
}*/

#[derive(Clone)]
pub struct UploadedTriMesh {
    pub index_buffer_offset: u64,
    pub index_count: u32,
}

pub struct RasterMeshesData<'a> {
    pub meshes: &'a [UploadedTriMesh],
    pub instances: &'a [crate::render_client::MeshInstance],
    pub vertex_buffer: Arc<Buffer>,
    pub bindless_descriptor_set: vk::DescriptorSet,
}

pub fn raster_meshes(
    rg: &mut RenderGraph,
    render_pass: Arc<RenderPass>,
    depth_img: &mut Handle<Image>,
    color_img: &mut Handle<Image>,
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
            .push_constants_bytes(16),
    );

    let meshes: Vec<UploadedTriMesh> = mesh_data.meshes.to_vec();
    let instances: Vec<crate::render_client::MeshInstance> = mesh_data.instances.to_vec();

    let depth_ref = pass.raster(depth_img, AccessType::DepthStencilAttachmentWrite);
    let color_ref = pass.raster(color_img, AccessType::ColorAttachmentWrite);

    let vertex_buffer = mesh_data.vertex_buffer.clone();
    let bindless_descriptor_set = mesh_data.bindless_descriptor_set;

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
                );

                pipeline.push_constants(
                    cb.raw,
                    vk::ShaderStageFlags::ALL_GRAPHICS,
                    0,
                    std::slice::from_raw_parts(&push_constants as *const _ as *const u8, 16),
                );

                raw_device.cmd_draw_indexed(cb.raw, mesh.index_count, 1, 0, 0, 0);
            }
        }

        api.end_render_pass();
    });
}

#[allow(clippy::too_many_arguments)]
pub fn light_gbuffer(
    rg: &mut RenderGraph,
    gbuffer: &Handle<Image>,
    depth: &Handle<Image>,
    sun_shadow_mask: &Handle<Image>,
    ssgi: &Handle<Image>,
    rtr: &Handle<Image>,
    base_light: &Handle<Image>,
    output: &mut Handle<Image>,
    debug_output: &mut Handle<Image>,
    csgi_volume: &csgi::CsgiVolume,
    csgi2_volume: &csgi2::Csgi2Volume,
    bindless_descriptor_set: vk::DescriptorSet,
) {
    SimpleRenderPass::new_compute(
        rg.add_pass("light gbuffer"),
        "/assets/shaders/light_gbuffer.hlsl",
    )
    .read(gbuffer)
    .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
    .read(sun_shadow_mask)
    .read(ssgi)
    .read(rtr)
    .read(base_light)
    .write(output)
    .write(debug_output)
    .read(&csgi_volume.cascade0)
    .read(&csgi2_volume.direct_cascade0)
    .read(&csgi2_volume.indirect_cascade0)
    .constants((
        gbuffer.desc().extent_inv_extent_2d(),
        CSGI_SLICE_DIRS,
        csgi_volume.volume_centers,
    ))
    .raw_descriptor_set(1, bindless_descriptor_set)
    .dispatch(gbuffer.desc().extent);
}

pub fn reference_path_trace(
    rg: &mut RenderGraph,
    output_img: &mut Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
    tlas: &Handle<RayTracingAcceleration>,
) {
    SimpleRenderPass::new_rt(
        rg.add_pass("reference pt"),
        "/assets/shaders/rt/reference_path_trace.rgen.hlsl",
        &[
            "/assets/shaders/rt/triangle.rmiss.hlsl",
            "/assets/shaders/rt/shadow.rmiss.hlsl",
        ],
        &["/assets/shaders/rt/triangle.rchit.hlsl"],
    )
    .write(output_img)
    .raw_descriptor_set(1, bindless_descriptor_set)
    .trace_rays(tlas, output_img.desc().extent);
}

pub fn normalize_accum(
    rg: &mut RenderGraph,
    input: &Handle<Image>,
    fmt: vk::Format,
    bindless_descriptor_set: vk::DescriptorSet,
) -> Handle<Image> {
    let mut output = rg.create((*input.desc()).format(fmt));

    SimpleRenderPass::new_compute(
        rg.add_pass("normalize accum"),
        "/assets/shaders/normalize_accum.hlsl",
    )
    .read(input)
    .write(&mut output)
    .raw_descriptor_set(1, bindless_descriptor_set)
    .dispatch(input.desc().extent);

    output
}

pub fn trace_sun_shadow_mask(
    rg: &mut RenderGraph,
    depth_img: &Handle<Image>,
    tlas: &Handle<RayTracingAcceleration>,
) -> Handle<Image> {
    let mut output_img = rg.create(depth_img.desc().format(vk::Format::R8_UNORM));

    SimpleRenderPass::new_rt(
        rg.add_pass("trace shadow mask"),
        "/assets/shaders/rt/trace_sun_shadow_mask.rgen.hlsl",
        &["/assets/shaders/rt/shadow.rmiss.hlsl"],
        &[],
    )
    .read_aspect(&depth_img, vk::ImageAspectFlags::DEPTH)
    .write(&mut output_img)
    .trace_rays(tlas, output_img.desc().extent);

    output_img
}

pub fn calculate_reprojection_map(rg: &mut RenderGraph, depth: &Handle<Image>) -> Handle<Image> {
    let mut output_tex = rg.create(depth.desc().format(vk::Format::R16G16B16A16_SFLOAT));

    SimpleRenderPass::new_compute(
        rg.add_pass("reprojection map"),
        "/assets/shaders/calculate_reprojection_map.hlsl",
    )
    .read_aspect(depth, vk::ImageAspectFlags::DEPTH)
    .write(&mut output_tex)
    .constants(output_tex.desc().extent_inv_extent_2d())
    .dispatch(output_tex.desc().extent);

    output_tex
}

fn blur_downsample_2x2(rg: &mut RenderGraph, input: &Handle<Image>) -> Handle<Image> {
    let mut output = rg.create(
        input
            .desc()
            .half_res()
            .format(vk::Format::B10G11R11_UFLOAT_PACK32),
    );

    SimpleRenderPass::new_compute(rg.add_pass("blur0"), "/assets/shaders/blur.hlsl")
        .read(input)
        .write(&mut output)
        .dispatch(output.desc().extent);

    output
}

pub fn blur_pyramid(rg: &mut RenderGraph, input: &Handle<Image>) -> Handle<Image> {
    let skip_n_bottom_mips = 1;
    let mut pyramid_desc = input
        .desc()
        .half_res()
        .format(vk::Format::B10G11R11_UFLOAT_PACK32) // R16G16B16A16_SFLOAT
        .all_mip_levels();
    pyramid_desc.mip_levels = (pyramid_desc
        .mip_levels
        .overflowing_sub(skip_n_bottom_mips)
        .0)
        .max(1);

    let mut output = rg.create(pyramid_desc);

    SimpleRenderPass::new_compute(rg.add_pass("blur0"), "/assets/shaders/blur.hlsl")
        .read(input)
        .write_view(
            &mut output,
            ImageViewDesc::builder()
                .base_mip_level(0)
                .level_count(Some(1)),
        )
        .dispatch(output.desc().extent);

    for target_mip in 1..(output.desc().mip_levels as u32) {
        let downsample_amount = 1 << target_mip;

        SimpleRenderPass::new_compute(
            rg.add_pass(&format!("blur{}", target_mip)),
            "/assets/shaders/blur.hlsl",
        )
        .read_view(
            &output,
            ImageViewDesc::builder()
                .base_mip_level(target_mip - 1)
                .level_count(Some(1)),
        )
        .write_view(
            &mut output,
            ImageViewDesc::builder()
                .base_mip_level(target_mip)
                .level_count(Some(1)),
        )
        .dispatch(
            output
                .desc()
                .div_extent([downsample_amount, downsample_amount, 1])
                .extent,
        );
    }

    output
}

pub fn rev_blur_pyramid(rg: &mut RenderGraph, in_pyramid: &Handle<Image>) -> Handle<Image> {
    let mut output = rg.create(*in_pyramid.desc());

    for target_mip in (0..(output.desc().mip_levels as u32 - 1)).rev() {
        let downsample_amount = 1 << target_mip;
        let output_extent: [u32; 3] = output
            .desc()
            .div_extent([downsample_amount, downsample_amount, 1])
            .extent;
        let src_mip: u32 = target_mip + 1;
        let self_weight = if src_mip == output.desc().mip_levels as u32 {
            0.0f32
        } else {
            0.5f32
        };

        SimpleRenderPass::new_compute(
            rg.add_pass(&format!("rev_blur{}", target_mip)),
            "/assets/shaders/rev_blur.hlsl",
        )
        .read_view(
            &in_pyramid,
            ImageViewDesc::builder()
                .base_mip_level(target_mip)
                .level_count(Some(1)),
        )
        .read_view(
            &output,
            ImageViewDesc::builder()
                .base_mip_level(src_mip)
                .level_count(Some(1)),
        )
        .write_view(
            &mut output,
            ImageViewDesc::builder()
                .base_mip_level(target_mip)
                .level_count(Some(1)),
        )
        .constants((output_extent[0], output_extent[1], src_mip, self_weight))
        .dispatch(output_extent);
    }

    output
}

pub fn post_process(
    rg: &mut RenderGraph,
    input: &Handle<Image>,
    bindless_descriptor_set: vk::DescriptorSet,
) -> Handle<Image> {
    let blur_pyramid = blur_pyramid(rg, input);
    let rev_blur_pyramid = rev_blur_pyramid(rg, &blur_pyramid);

    let mut output = rg.create(input.desc().format(vk::Format::B10G11R11_UFLOAT_PACK32));
    let blur_pyramid_mip_count: u32 = blur_pyramid.desc().mip_levels as _;

    SimpleRenderPass::new_compute(rg.add_pass("glare"), "/assets/shaders/glare.hlsl")
        .read(input)
        .read(&blur_pyramid)
        .read(&rev_blur_pyramid)
        .write(&mut output)
        .raw_descriptor_set(1, bindless_descriptor_set)
        .constants((output.desc().extent_inv_extent_2d(), blur_pyramid_mip_count))
        .dispatch(output.desc().extent);

    output
}
