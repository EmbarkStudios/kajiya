use crate::{bytes::as_byte_slice, vulkan};
use ash::vk;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::mem::{align_of, size_of};
use vulkan::buffer::Buffer;

pub const DYNAMIC_CONSTANTS_SIZE_BYTES: usize = 1024 * 1024 * 16;
pub const DYNAMIC_CONSTANTS_BUFFER_COUNT: usize = 2;

// Generally supported minimum uniform buffer size across vendors (maxUniformBufferRange)
// Could be bumped to 65536 if needed.
pub const MAX_DYNAMIC_CONSTANTS_BYTES_PER_DISPATCH: usize = 16384;

// TODO: Must be >= `minUniformBufferOffsetAlignment`. In practice <= 256.
pub const DYNAMIC_CONSTANTS_ALIGNMENT: usize = 256;

// Sadly we can't have unsized dynamic storage buffers sub-allocated from dynamic constants because WHOLE_SIZE blows up.
// https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2846#issuecomment-851744837
// For now, just a max size.
pub const MAX_DYNAMIC_CONSTANTS_STORAGE_BUFFER_BYTES: usize = 1024 * 1024;

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
        self.frame_parity = (self.frame_parity + 1) % DYNAMIC_CONSTANTS_BUFFER_COUNT;
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

    pub fn push_from_iter<T: Copy, Iter: Iterator<Item = T>>(&mut self, iter: Iter) -> u32 {
        let t_size = size_of::<T>();
        let t_align = align_of::<T>();

        assert!(self.frame_offset_bytes + t_size < DYNAMIC_CONSTANTS_SIZE_BYTES);
        assert!(DYNAMIC_CONSTANTS_ALIGNMENT % t_align == 0);

        let buffer_offset = self.current_offset() as usize;
        assert!(buffer_offset % t_align == 0);

        let mut dst_offset = buffer_offset;
        for t in iter {
            let dst = &mut self.buffer.allocation.mapped_slice_mut().unwrap()
                [dst_offset..dst_offset + t_size];
            dst.copy_from_slice(as_byte_slice(&t));
            dst_offset += t_size + t_align - 1;
            dst_offset &= !(t_align - 1);
        }

        self.frame_offset_bytes += dst_offset - buffer_offset;
        self.frame_offset_bytes += DYNAMIC_CONSTANTS_ALIGNMENT - 1;
        self.frame_offset_bytes &= !(DYNAMIC_CONSTANTS_ALIGNMENT - 1);

        buffer_offset as _
    }
}
