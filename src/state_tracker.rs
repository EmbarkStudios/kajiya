use crate::backend::{
    barrier::{record_image_barrier, ImageBarrier},
    device::Device,
};
use ash::vk;

pub struct LocalImageStateTracker<'device> {
    resource: vk::Image,
    aspect_mask: vk::ImageAspectFlags,
    current_state: vk_sync::AccessType,
    cb: vk::CommandBuffer,
    device: &'device Device,
}

impl<'device> LocalImageStateTracker<'device> {
    pub fn new(
        resource: vk::Image,
        aspect_mask: vk::ImageAspectFlags,
        current_state: vk_sync::AccessType,
        cb: vk::CommandBuffer,
        device: &'device Device,
    ) -> Self {
        Self {
            resource,
            aspect_mask,
            current_state,
            cb,
            device,
        }
    }

    pub fn transition(&mut self, state: vk_sync::AccessType) {
        record_image_barrier(
            &self.device,
            self.cb,
            ImageBarrier::new(self.resource, self.current_state, state, self.aspect_mask),
        );

        self.current_state = state;
    }
}
