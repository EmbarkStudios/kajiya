use crate::rg::*;

pub fn synth_gradients(rg: &mut RenderGraph, desc: ImageDesc) -> Handle<Image> {
    let mut pass = rg.add_pass();

    let pipeline = pass.register_compute_pipeline("/assets/shaders/gradients.hlsl");

    let mut output = pass.create(&desc);
    let output_ref = pass.write(&mut output, AccessType::ComputeShaderWrite);

    pass.render(move |api| {
        let pipeline = api.bind_compute_pipeline(
            pipeline
                .into_binding()
                .descriptor_set(0, &[output_ref.bind(ImageViewDescBuilder::default())]),
        );

        pipeline.dispatch(desc.extent);
    });

    output
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
