use kajiya_backend::{ash::vk, vulkan, BackendError};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::ops::Range;
use vulkan::buffer::{Buffer, BufferDesc};

pub trait BufferDataSource {
    fn as_bytes(&self) -> &[u8];
    fn alignment(&self) -> u64;
}

struct PendingBufferUpload {
    source: Box<dyn BufferDataSource>,
    offset: u64,
}

impl<T: Copy> BufferDataSource for &'static [T] {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const u8,
                self.len() * std::mem::size_of::<T>(),
            )
        }
    }

    fn alignment(&self) -> u64 {
        std::mem::align_of::<T>() as u64
    }
}

impl<T: Copy> BufferDataSource for Vec<T> {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const u8,
                self.len() * std::mem::size_of::<T>(),
            )
        }
    }

    fn alignment(&self) -> u64 {
        std::mem::align_of::<T>() as u64
    }
}
pub struct BufferBuilder {
    //buf_slice: &'a mut [u8],
    pending_uploads: Vec<PendingBufferUpload>,
    current_offset: u64,
}

impl Default for BufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferBuilder {
    pub fn new() -> Self {
        Self {
            pending_uploads: Vec::new(),
            current_offset: 0,
        }
    }

    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }

    pub fn append(&mut self, data: impl BufferDataSource + 'static) -> u64 {
        let alignment = data.alignment();
        assert!(alignment.count_ones() == 1);

        let data_start = (self.current_offset + alignment - 1) & !(alignment - 1);
        let data_len = data.as_bytes().len() as u64;

        self.pending_uploads.push(PendingBufferUpload {
            source: Box::new(data),
            offset: data_start,
        });
        self.current_offset = data_start + data_len;

        data_start
    }

    pub fn upload(
        self,
        device: &kajiya_backend::Device,
        target: &mut Buffer,
        target_offset: u64,
    ) -> Result<(), BackendError> {
        assert!(
            self.pending_uploads
                .iter()
                .map(|chunk| chunk.source.as_bytes().len())
                .sum::<usize>()
                + target_offset as usize
                <= target.desc.size
        );
        let target = target.raw;

        // TODO: share a common staging buffer, don't leak
        const STAGING_BYTES: usize = 16 * 1024 * 1024;
        let mut staging_buffer = device.create_buffer(
            BufferDesc::new_cpu_to_gpu(STAGING_BYTES, vk::BufferUsageFlags::TRANSFER_SRC),
            "BufferBuilder staging",
            None,
        )?;

        struct UploadChunk {
            pending_idx: usize,
            src_range: Range<usize>,
        }

        // TODO: merge chunks to perform fewer uploads if multiple source regions fit in one chunk
        let chunks: Vec<UploadChunk> = self
            .pending_uploads
            .iter()
            .enumerate()
            .flat_map(|(pending_idx, pending)| {
                let byte_count = pending.source.as_bytes().len();
                let chunk_count = (byte_count + STAGING_BYTES - 1) / STAGING_BYTES;
                (0..chunk_count).map(move |chunk| UploadChunk {
                    pending_idx,
                    src_range: chunk * STAGING_BYTES..((chunk + 1) * STAGING_BYTES).min(byte_count),
                })
            })
            .collect();

        for UploadChunk {
            pending_idx,
            src_range,
        } in chunks
        {
            let pending = &self.pending_uploads[pending_idx];
            staging_buffer.allocation.mapped_slice_mut().unwrap()
                [0..(src_range.end - src_range.start)]
                .copy_from_slice(&pending.source.as_bytes()[src_range.start..src_range.end]);

            device.with_setup_cb(|cb| unsafe {
                device.raw.cmd_copy_buffer(
                    cb,
                    staging_buffer.raw,
                    target,
                    &[vk::BufferCopy::builder()
                        .src_offset(0u64)
                        .dst_offset(target_offset + pending.offset + src_range.start as u64)
                        .size((src_range.end - src_range.start) as u64)
                        .build()],
                );
            })?;
        }

        Ok(())
    }
}
