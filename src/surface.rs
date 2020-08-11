use anyhow::Result;
use ash::{extensions::khr, vk};

pub struct Surface {
    pub(crate) raw: vk::SurfaceKHR,
    pub(crate) fns: khr::Surface,
}

impl Surface {
    pub fn new(
        instance: &crate::instance::Instance,
        window: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Self> {
        let surface =
            unsafe { ash_window::create_surface(&instance.entry, &instance.raw, window, None)? };
        let surface_loader = khr::Surface::new(&instance.entry, &instance.raw);

        Ok(Self {
            raw: surface,
            fns: surface_loader,
        })
    }
}
