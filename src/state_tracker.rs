use crate::backend::barrier::{record_image_barrier, ImageBarrier};
use ash::vk;

pub struct LocalImageStateTracker<'device> {
    resource: vk::Image,
    current_state: vk_sync::AccessType,
    cb: vk::CommandBuffer,
    device: &'device ash::Device,
}

impl<'device> LocalImageStateTracker<'device> {
    pub fn new(
        resource: vk::Image,
        current_state: vk_sync::AccessType,
        cb: vk::CommandBuffer,
        device: &'device ash::Device,
    ) -> Self {
        Self {
            resource,
            current_state,
            cb,
            device,
        }
    }

    pub fn transition(&mut self, state: vk_sync::AccessType) {
        record_image_barrier(
            &self.device,
            self.cb,
            ImageBarrier::new(self.resource, self.current_state, state),
        );

        self.current_state = state;
    }
}
