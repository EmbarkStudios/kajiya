#![allow(dead_code)]
use std::mem::size_of;

use kajiya_backend::{
    ash::vk,
    vulkan::buffer::{Buffer, BufferDesc},
};
use kajiya_rg::{self as rg, SimpleRenderPass};

pub fn inclusive_prefix_scan_u32_1m(rg: &mut rg::RenderGraph, input_buf: &mut rg::Handle<Buffer>) {
    const SEGMENT_SIZE: usize = 1024;

    SimpleRenderPass::new_compute(
        rg.add_pass("_prefix scan 1"),
        "/shaders/prefix_scan/inclusive_prefix_scan.hlsl",
    )
    .write(input_buf)
    .dispatch([(SEGMENT_SIZE * SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect

    let mut segment_sum_buf = rg.create(BufferDesc::new_gpu_only(
        size_of::<u32>() * SEGMENT_SIZE,
        vk::BufferUsageFlags::empty(),
    ));
    SimpleRenderPass::new_compute(
        rg.add_pass("_prefix scan 2"),
        "/shaders/prefix_scan/inclusive_prefix_scan_segments.hlsl",
    )
    .read(input_buf)
    .write(&mut segment_sum_buf)
    .dispatch([(SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect

    SimpleRenderPass::new_compute(
        rg.add_pass("_prefix scan merge"),
        "/shaders/prefix_scan/inclusive_prefix_scan_merge.hlsl",
    )
    .write(input_buf)
    .read(&segment_sum_buf)
    .dispatch([(SEGMENT_SIZE * SEGMENT_SIZE / 2) as u32, 1, 1]); // TODO: indirect
}
