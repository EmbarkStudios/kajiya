use anyhow::Result;
use ash::{extensions::khr, vk};
use std::sync::Arc;

pub struct Surface {
    pub(crate) raw: vk::SurfaceKHR,
    pub(crate) fns: khr::Surface,
}

impl Surface {
    pub fn create(
        instance: &crate::instance::Instance,
        window: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Arc<Self>> {
        let surface =
            unsafe { ash_window::create_surface(&instance.entry, &instance.raw, window, None)? };
        let surface_loader = khr::Surface::new(&instance.entry, &instance.raw);

        Ok(Arc::new(Self {
            raw: surface,
            fns: surface_loader,
        }))
    }
}
