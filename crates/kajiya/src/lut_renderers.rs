use kajiya_backend::{ash::vk, vk_sync::AccessType, ImageDesc};
use kajiya_rg::{BindRgRef, IntoRenderPassPipelineBinding};

#[allow(unused_imports)]
use kajiya_backend::{ash::vk::ImageUsageFlags, vulkan::image::*};

use crate::image_lut::ComputeImageLut;

pub struct BrdfFgLutComputer;

impl ComputeImageLut for BrdfFgLutComputer {
    fn create(&mut self, device: &kajiya_backend::Device) -> kajiya_backend::Image {
        device
            .create_image(
                ImageDesc::new_2d(vk::Format::R16G16_SFLOAT, [64, 64])
                    .usage(ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED),
                vec![],
            )
            .expect("image")
    }

    fn compute(
        &mut self,
        rg: &mut kajiya_rg::RenderGraph,
        img: &mut kajiya_rg::Handle<kajiya_backend::Image>,
    ) {
        let mut pass = rg.add_pass("brdf_fg lut");

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
