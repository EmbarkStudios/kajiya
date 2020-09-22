use std::sync::Arc;

use ash::{version::DeviceV1_0, vk};

use crate::{backend::image::ImageViewDesc, backend::shader::*, rg::*};

pub fn create_image(rg: &mut RenderGraph, desc: ImageDesc) -> Handle<Image> {
    let mut pass = rg.add_pass();
    pass.create(&desc)
}

pub fn clear_depth(rg: &mut RenderGraph, img: &mut Handle<Image>) {
    let mut pass = rg.add_pass();
    let output_ref = pass.write(img, AccessType::TransferWrite);

    pass.render(move |api| {
        let raw_device = &api.device().raw;
        let cb = api.cb;

        let image = api.resources.image(output_ref);

        unsafe {
            raw_device.cmd_clear_depth_stencil_image(
                cb.raw,
                image.raw,
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &ash::vk::ClearDepthStencilValue {
                    depth: 0f32,
                    stencil: 0,
                },
                std::slice::from_ref(&vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                    level_count: 1 as u32,
                    layer_count: 1,
                    ..Default::default()
                }),
            );
        }
    });
}

pub fn raymarch_sdf(
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

pub fn raster_sdf(
    rg: &mut RenderGraph,
    render_pass: Arc<RenderPass>,
    depth_img: &mut Handle<Image>,
    color_img: &mut Handle<Image>,
    sdf_img: &Handle<Image>,
    brick_inst_buffer: &Handle<Buffer>,
    brick_meta_buffer: &Handle<Buffer>,
    cube_index_buffer: &Handle<Buffer>,
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
        &RasterPipelineDesc::builder()
            .render_pass(render_pass.clone())
            .face_cull(true),
    );

    let sdf_ref = pass.read(
        sdf_img,
        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
    );
    let brick_inst_buffer = pass.read(
        brick_inst_buffer,
        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
    );
    let brick_meta_buffer = pass.read(brick_meta_buffer, AccessType::IndirectBuffer);
    let cube_index_buffer = pass.read(cube_index_buffer, AccessType::IndexBuffer);

    let depth_ref = pass.raster(depth_img, AccessType::DepthStencilAttachmentWrite);
    let color_ref = pass.raster(color_img, AccessType::ColorAttachmentWrite);

    pass.render(move |api| {
        let [width, height, _] = color_ref.desc.extent;

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

            // TODO: dispatch indirect. just one draw, but with many instances.
            //raw_device.cmd_draw_indexed_indirect();
        }

        api.end_render_pass();
    });
}

pub fn blur(rg: &mut RenderGraph, input: &Handle<Image>) -> Handle<Image> {
    let mut pass = rg.add_pass();

    let pipeline = pass.register_compute_pipeline("/assets/shaders/blur.hlsl");

    let input_ref = pass.read(
        input,
        AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
    );
    let mut output = pass.create(input.desc());
    let output_ref = pass.write(&mut output, AccessType::ComputeShaderWrite);

    pass.render(move |api| {
        let pipeline = api.bind_compute_pipeline(pipeline.into_binding().descriptor_set(
            0,
            &[
                input_ref.bind(ImageViewDescBuilder::default()),
                output_ref.bind(ImageViewDescBuilder::default()),
            ],
        ));

        pipeline.dispatch(input_ref.desc.extent);
    });

    output
}
