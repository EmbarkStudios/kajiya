use ash::{version::DeviceV1_0, vk};

pub struct ImageBarrier {
    image: vk::Image,
    prev_access: vk_sync::AccessType,
    next_access: vk_sync::AccessType,
    discard: bool,
}

pub fn record_image_barrier(device: &ash::Device, cb: vk::CommandBuffer, barrier: ImageBarrier) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    vk_sync::cmd::pipeline_barrier(
        device.fp_v1_0(),
        cb,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[barrier.prev_access],
            next_accesses: &[barrier.next_access],
            previous_layout: vk_sync::ImageLayout::Optimal,
            next_layout: vk_sync::ImageLayout::Optimal,
            discard_contents: barrier.discard,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: barrier.image,
            range,
        }],
    );
}

impl ImageBarrier {
    pub fn new(
        image: vk::Image,
        prev_access: vk_sync::AccessType,
        next_access: vk_sync::AccessType,
    ) -> Self {
        Self {
            image,
            prev_access,
            next_access,
            discard: false,
        }
    }

    pub fn with_discard(mut self, discard: bool) -> Self {
        self.discard = discard;
        self
    }
}
