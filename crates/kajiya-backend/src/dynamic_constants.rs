use crate::bytes::as_byte_slice;
use crate::vulkan;
use ash::vk;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::mem::size_of;
use vulkan::buffer::Buffer;

pub const DYNAMIC_CONSTANTS_SIZE_BYTES: usize = 1024 * 1024 * 16;
pub const DYNAMIC_CONSTANTS_ALIGNMENT: usize = 64;

pub struct DynamicConstants {
    pub buffer: Buffer,
    frame_offset_bytes: usize,
    frame_parity: usize,
}

impl DynamicConstants {
    pub fn new(buffer: Buffer) -> Self {
        Self {
            buffer,
            frame_offset_bytes: 0,
            frame_parity: 0,
        }
    }

    pub fn advance_frame(&mut self) {
        self.frame_parity = 1 - self.frame_parity;
        self.frame_offset_bytes = 0;
    }

    pub fn current_offset(&self) -> u32 {
        (self.frame_parity * DYNAMIC_CONSTANTS_SIZE_BYTES + self.frame_offset_bytes) as u32
    }

    pub fn current_device_address(&self, device: &crate::Device) -> vk::DeviceAddress {
        self.buffer.device_address(device) + self.current_offset() as vk::DeviceAddress
    }

    pub fn push<T: Copy>(&mut self, t: &T) -> u32 {
        let t_size = size_of::<T>();
        assert!(self.frame_offset_bytes + t_size < DYNAMIC_CONSTANTS_SIZE_BYTES);

        let buffer_offset = self.current_offset() as usize;
        let dst = &mut self.buffer.allocation.mapped_slice_mut().unwrap()
            [buffer_offset..buffer_offset + t_size];

        dst.copy_from_slice(as_byte_slice(t));

        let t_size_aligned =
            (t_size + DYNAMIC_CONSTANTS_ALIGNMENT - 1) & !(DYNAMIC_CONSTANTS_ALIGNMENT - 1);
        self.frame_offset_bytes += t_size_aligned;

        buffer_offset as _
    }
}
