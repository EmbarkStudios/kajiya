use slingshot::{
    ash::vk,
    rg::{BindRgRef, IntoRenderPassPipelineBinding},
    vk_sync::AccessType,
    ImageDesc,
};
use slingshot::{ash::vk::ImageUsageFlags, ImageViewDescBuilder};

use crate::image_lut::ComputeImageLut;

pub struct BrdfFgLutComputer;

impl ComputeImageLut for BrdfFgLutComputer {
    fn create(&mut self, device: &slingshot::Device) -> slingshot::Image {
        device
            .create_image(
                ImageDesc::new_2d(vk::Format::R16G16_SFLOAT, [64, 64])
                    .usage(ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED),
                None,
            )
            .expect("image")
    }

    fn compute(
        &mut self,
        rg: &mut slingshot::rg::RenderGraph,
        img: &mut slingshot::rg::Handle<slingshot::Image>,
    ) {
        let mut pass = rg.add_pass();

        let pipeline = pass.register_compute_pipeline("/assets/shaders/lut/brdf_fg.hlsl");
        let img_ref = pass.write(img, AccessType::ComputeShaderWrite);

        pass.render(move |api| {
            let pipeline = api.bind_compute_pipeline(
                pipeline.into_binding().descriptor_set(0, &[img_ref.bind()]),
            );

            pipeline.dispatch(img_ref.desc().extent);
        });
    }
}
