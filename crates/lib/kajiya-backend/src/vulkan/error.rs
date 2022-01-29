use std::collections::HashMap;

use crate::{BackendError, Device};

use super::device::CommandBuffer;

#[derive(Default)]
pub(crate) struct CrashMarkerNames {
    next_idx: u32,
    names: HashMap<u32, (u32, String)>,
}

impl CrashMarkerNames {
    fn insert_name(&mut self, name: String) -> u32 {
        // TODO: retire those with frames
        let idx = self.next_idx;
        let small_idx = idx % 4096;

        self.next_idx = self.next_idx.wrapping_add(1);
        self.names.insert(small_idx, (idx, name));

        idx
    }

    fn get_name(&self, marker: u32) -> Option<&str> {
        match self.names.get(&(marker % 4096)) {
            Some((last_marker_idx, last_marker_str)) if *last_marker_idx == marker => {
                Some(last_marker_str)
            }
            _ => None,
        }
    }
}

impl Device {
    pub fn record_crash_marker(&self, cb: &CommandBuffer, name: String) {
        let mut names = self.crash_marker_names.lock();
        let idx = names.insert_name(name);

        unsafe {
            self.raw
                .cmd_fill_buffer(cb.raw, self.crash_tracking_buffer.raw, 0, 4, idx);
        }
    }

    pub fn report_error(&self, err: BackendError) -> never::Never {
        match err {
            BackendError::Vulkan {
                err: ash::vk::Result::ERROR_DEVICE_LOST,
                ..
            } => {
                // Something went very wrong. Find the last marker which was successfully written
                // to the crash tracking buffer, and report its corresponding name.
                let last_marker = self
                    .crash_tracking_buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *const u32;
                let last_marker: u32 = unsafe { *last_marker.as_ref().unwrap() };

                let names = self.crash_marker_names.lock();
                let panic_msg = match names.get_name(last_marker) {
                    Some(last_marker_str) => {
                        format!(
                            "Queue submit failed. Last marker: {} => {}",
                            last_marker, last_marker_str
                        )
                    }
                    _ => {
                        format!("Queue submit failed. Last marker: {}", last_marker)
                    }
                };

                log::error!("{}", panic_msg);
                panic!("{}", panic_msg);
            }
            _ => panic!("Kajiya backend error: {:?}", err),
        }
    }
}
