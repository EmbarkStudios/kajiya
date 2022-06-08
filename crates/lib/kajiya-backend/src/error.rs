use backtrace::Backtrace as Bt;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Allocation failed for {name:?}: {inner:?}")]
    Allocation {
        inner: gpu_allocator::AllocationError,
        name: String,
    },

    #[error("Vulkan error: {err:?}; {trace:?}")]
    Vulkan { err: ash::vk::Result, trace: Bt },

    #[error("Invalid resource access: {info:?}")]
    ResourceAccess { info: String },
}

impl From<ash::vk::Result> for BackendError {
    fn from(err: ash::vk::Result) -> Self {
        Self::Vulkan {
            err,
            trace: Bt::new(),
        }
    }
}
