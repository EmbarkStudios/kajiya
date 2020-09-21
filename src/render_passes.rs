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
