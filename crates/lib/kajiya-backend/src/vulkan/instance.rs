use anyhow::Result;
use ash::{extensions::ext, vk};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
    sync::Arc,
};

#[derive(Default)]
pub struct DeviceBuilder {
    pub required_extensions: Vec<&'static CStr>,
    pub graphics_debugging: bool,
}

impl DeviceBuilder {
    pub fn build(self) -> Result<Arc<Instance>> {
        Ok(Arc::new(Instance::create(self)?))
    }

    pub fn required_extensions(mut self, required_extensions: Vec<&'static CStr>) -> Self {
        self.required_extensions = required_extensions;
        self
    }

    pub fn graphics_debugging(mut self, graphics_debugging: bool) -> Self {
        self.graphics_debugging = graphics_debugging;
        self
    }
}

pub struct Instance {
    pub(crate) entry: ash::Entry,
    pub raw: ash::Instance,
    #[allow(dead_code)]
    pub(crate) debug_callback: Option<vk::DebugReportCallbackEXT>,
    #[allow(dead_code)]
    #[allow(deprecated)]
    pub(crate) debug_loader: Option<ext::DebugReport>,
    pub(crate) debug_utils: Option<ash::extensions::ext::DebugUtils>,
}

impl Instance {
    pub fn builder() -> DeviceBuilder {
        DeviceBuilder::default()
    }

    fn extension_names(builder: &DeviceBuilder) -> Vec<*const i8> {
        let mut names = vec![vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr()];

        if builder.graphics_debugging {
            #[allow(deprecated)]
            names.push(ext::DebugReport::name().as_ptr());
            names.push(vk::ExtDebugUtilsFn::name().as_ptr());
        }

        names
    }

    fn layer_names(builder: &DeviceBuilder) -> Vec<CString> {
        let mut layer_names = Vec::new();
        if builder.graphics_debugging {
            layer_names.push(CString::new("VK_LAYER_KHRONOS_validation").unwrap());
        }
        layer_names
    }

    fn create(builder: DeviceBuilder) -> Result<Self> {
        let entry = unsafe { ash::Entry::new()? };
        let instance_extensions = builder
            .required_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .chain(Self::extension_names(&builder).into_iter())
            .collect::<Vec<_>>();

        let layer_names = Self::layer_names(&builder);
        let layer_names: Vec<*const i8> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let app_desc = vk::ApplicationInfo::builder().api_version(vk::make_api_version(0, 1, 2, 0));

        let instance_desc = vk::InstanceCreateInfo::builder()
            .application_info(&app_desc)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe { entry.create_instance(&instance_desc, None)? };
        info!("Created a Vulkan instance");

        let (debug_loader, debug_callback, debug_utils) = if builder.graphics_debugging {
            let debug_info = ash::vk::DebugReportCallbackCreateInfoEXT {
                flags: ash::vk::DebugReportFlagsEXT::ERROR
                    | ash::vk::DebugReportFlagsEXT::WARNING
                    | ash::vk::DebugReportFlagsEXT::PERFORMANCE_WARNING,
                pfn_callback: Some(vulkan_debug_callback),
                ..Default::default()
            };

            #[allow(deprecated)]
            let debug_loader = ext::DebugReport::new(&entry, &instance);

            let debug_callback = unsafe {
                #[allow(deprecated)]
                debug_loader
                    .create_debug_report_callback(&debug_info, None)
                    .unwrap()
            };

            let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);

            (Some(debug_loader), Some(debug_callback), Some(debug_utils))
        } else {
            (None, None, None)
        };

        Ok(Self {
            entry,
            raw: instance,
            debug_callback,
            debug_loader,
            debug_utils,
        })
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    _flags: vk::DebugReportFlagsEXT,
    _obj_type: vk::DebugReportObjectTypeEXT,
    _src_obj: u64,
    _location: usize,
    _msg_code: i32,
    _layer_prefix: *const c_char,
    message: *const c_char,
    _user_data: *mut c_void,
) -> u32 {
    let message = CStr::from_ptr(message).to_str().unwrap();

    #[allow(clippy::if_same_then_else)]
    if message.starts_with("Validation Error: [ VUID-VkWriteDescriptorSet-descriptorType-00322")
        || message.starts_with("Validation Error: [ VUID-VkWriteDescriptorSet-descriptorType-02752")
    {
        // Validation layers incorrectly report an error in pushing immutable sampler descriptors.
        //
        // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushDescriptorSetKHR.html
        // This documentation claims that it's necessary to push immutable samplers.
    } else if message.starts_with("Validation Performance Warning") {
    } else if message.starts_with("Validation Warning: [ VUID_Undefined ]") {
        log::warn!("{}\n", message);
    } else {
        log::error!("{}\n", message);
    }

    ash::vk::FALSE
}
