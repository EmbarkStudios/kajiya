use std::sync::Arc;

use kajiya_backend::{ash::vk, vk_sync::AccessType, vulkan::image::*, BackendError, Device};
use kajiya_rg::{self as rg};
use rg::{Buffer, BufferDesc, RenderGraph, SimpleRenderPass};

use crate::world_renderer::HistogramClipping;

pub fn blur_pyramid(rg: &mut RenderGraph, input: &rg::Handle<Image>) -> rg::Handle<Image> {
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

    SimpleRenderPass::new_compute_rust(rg.add_pass("_blur0"), "blur::blur_cs")
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
            rg.add_pass(&format!("_blur{}", target_mip)),
            "/shaders/blur.hlsl",
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

pub fn rev_blur_pyramid(rg: &mut RenderGraph, in_pyramid: &rg::Handle<Image>) -> rg::Handle<Image> {
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

        SimpleRenderPass::new_compute_rust(
            rg.add_pass(&format!("_rev_blur{}", target_mip)),
            "rev_blur::rev_blur_cs",
        )
        .read_view(
            in_pyramid,
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
        .constants((output_extent[0], output_extent[1], self_weight))
        .dispatch(output_extent);
    }

    output
}

const LUMINANCE_HISTOGRAM_BIN_COUNT: usize = 256;
const LUMINANCE_HISTOGRAM_MIN_LOG2: f64 = -16.0;
const LUMINANCE_HISTOGRAM_MAX_LOG2: f64 = 16.0;

pub struct PostProcessRenderer {
    histogram_buffer: Arc<Buffer>,
    pub image_log2_lum: f32,
}

impl PostProcessRenderer {
    pub fn new(device: &Device) -> Result<Self, BackendError> {
        Ok(Self {
            histogram_buffer: Arc::new(device.create_buffer(
                BufferDesc::new_gpu_to_cpu(
                    std::mem::size_of::<u32>() * LUMINANCE_HISTOGRAM_BIN_COUNT,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                "luminance histogram",
                None,
            )?),
            image_log2_lum: 0.0,
        })
    }

    fn calculate_luminance_histogram(
        &mut self,
        rg: &mut RenderGraph,
        blur_pyramid: &rg::Handle<Image>,
    ) -> rg::Handle<Buffer> {
        let mut tmp_histogram = rg.create(BufferDesc::new_gpu_only(
            std::mem::size_of::<u32>() * LUMINANCE_HISTOGRAM_BIN_COUNT,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        ));

        // Start with input downsampled to a fairly consistent size.
        let input_mip_level: u32 = blur_pyramid.desc().mip_levels.saturating_sub(7) as u32;

        let mip_extent = blur_pyramid
            .desc()
            .div_up_extent([1 << input_mip_level, 1 << input_mip_level, 1])
            .extent;

        SimpleRenderPass::new_compute(
            rg.add_pass("_clear histogram"),
            "/shaders/post/luminance_histogram_clear.hlsl",
        )
        .write(&mut tmp_histogram)
        .dispatch([LUMINANCE_HISTOGRAM_BIN_COUNT as u32, 1, 1]);

        SimpleRenderPass::new_compute(
            rg.add_pass("calculate histogram"),
            "/shaders/post/luminance_histogram_calculate.hlsl",
        )
        .read_view(
            blur_pyramid,
            ImageViewDesc::builder()
                .base_mip_level(input_mip_level)
                .level_count(Some(1)),
        )
        .write(&mut tmp_histogram)
        .constants([mip_extent[0], mip_extent[1]])
        .dispatch(mip_extent);

        let mut dst_histogram = rg.import(self.histogram_buffer.clone(), AccessType::Nothing);
        SimpleRenderPass::new_compute(
            rg.add_pass("_copy histogram"),
            "/shaders/post/luminance_histogram_copy.hlsl",
        )
        .read(&tmp_histogram)
        .write(&mut dst_histogram)
        .dispatch([LUMINANCE_HISTOGRAM_BIN_COUNT as u32, 1, 1]);

        tmp_histogram
    }

    fn read_back_histogram(&mut self, exposure_histogram_clipping: HistogramClipping) {
        let mut histogram = [0u32; LUMINANCE_HISTOGRAM_BIN_COUNT];
        {
            let src = if let Some(src) = self.histogram_buffer.allocation.mapped_slice() {
                bytemuck::checked::cast_slice::<u8, u32>(src)
            } else {
                return;
            };

            histogram.copy_from_slice(src);
        }

        // Reject this much from the bottom and top end
        let outlier_frac_lo: f64 = exposure_histogram_clipping.low.min(1.0) as f64;
        let outlier_frac_hi: f64 =
            (exposure_histogram_clipping.high as f64).min(1.0 - outlier_frac_lo);

        let total_entry_count: u32 = histogram.iter().copied().sum();
        let reject_lo_entry_count = (total_entry_count as f64 * outlier_frac_lo) as u32;
        let entry_count_to_use =
            (total_entry_count as f64 * (1.0 - outlier_frac_lo - outlier_frac_hi)) as u32;

        let mut sum = 0.0;
        let mut used_count = 0;

        let mut left_to_reject = reject_lo_entry_count;
        let mut left_to_use = entry_count_to_use;

        for (bin_idx, count) in histogram.into_iter().enumerate() {
            let t = (bin_idx as f64 + 0.5) / LUMINANCE_HISTOGRAM_BIN_COUNT as f64;

            let count_to_use = count.saturating_sub(left_to_reject).min(left_to_use);
            left_to_reject = left_to_reject.saturating_sub(count);
            left_to_use = left_to_use.saturating_sub(count_to_use);

            sum += t * count_to_use as f64;
            used_count += count_to_use;
        }

        assert_eq!(entry_count_to_use, used_count);

        let mean = sum / used_count.max(1) as f64;
        self.image_log2_lum = (LUMINANCE_HISTOGRAM_MIN_LOG2
            + mean * (LUMINANCE_HISTOGRAM_MAX_LOG2 - LUMINANCE_HISTOGRAM_MIN_LOG2))
            as f32;

        // log::info!("mean log lum: {}", self.image_log2_lum);
    }

    pub fn render(
        &mut self,
        rg: &mut RenderGraph,
        input: &rg::Handle<Image>,
        //debug_input: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        post_exposure_mult: f32,
        contrast: f32,
        exposure_histogram_clipping: HistogramClipping,
    ) -> rg::Handle<Image> {
        self.read_back_histogram(exposure_histogram_clipping);

        let blur_pyramid = blur_pyramid(rg, input);
        let histogram = self.calculate_luminance_histogram(rg, &blur_pyramid);

        let rev_blur_pyramid = rev_blur_pyramid(rg, &blur_pyramid);

        let mut output = rg.create(input.desc().format(vk::Format::B10G11R11_UFLOAT_PACK32));

        //let blurred_luminance = edge_preserving_filter_luminance(rg, input);

        SimpleRenderPass::new_compute(rg.add_pass("post combine"), "/shaders/post_combine.hlsl")
            .read(input)
            //.read(debug_input)
            .read(&blur_pyramid)
            .read(&rev_blur_pyramid)
            .read(&histogram)
            //.read(&blurred_luminance)
            .write(&mut output)
            .raw_descriptor_set(1, bindless_descriptor_set)
            .constants((
                output.desc().extent_inv_extent_2d(),
                post_exposure_mult,
                contrast,
            ))
            .dispatch(output.desc().extent);

        output
    }
}
