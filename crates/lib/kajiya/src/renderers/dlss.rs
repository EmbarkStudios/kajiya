use std::{intrinsics::transmute, ptr};

use glam::Vec2;
use kajiya_backend::{ash::vk, vk_sync::AccessType, vulkan::image::*, RenderBackend};
use kajiya_rg::{self as rg};
use ngx_dlss::*;
use wchar::wchz;

pub struct DlssRenderer {
    dlss_feature: *mut NVSDK_NGX_Handle,
    ngx_params: *mut NVSDK_NGX_Parameter,
    pub current_supersample_offset: Vec2,
    frame_idx: u32,
}

macro_rules! ngx_checked {
    ($($t:tt)*) => {
        assert_eq!(NVSDK_NGX_Result_NVSDK_NGX_Result_Success, $($t)*)
    };
}

impl DlssRenderer {
    pub fn new(
        backend: &RenderBackend,
        input_resolution: [u32; 2],
        target_resolution: [u32; 2],
    ) -> Self {
        unsafe {
            let mut inst_ext_count = 0;
            let mut inst_exts = ptr::null_mut();
            let mut device_ext_count = 0;
            let mut device_exts = ptr::null_mut();

            assert_eq!(
                NVSDK_NGX_VULKAN_RequiredExtensions(
                    &mut inst_ext_count,
                    &mut inst_exts,
                    &mut device_ext_count,
                    &mut device_exts
                ),
                NVSDK_NGX_Result_NVSDK_NGX_Result_Success
            );

            /*let inst_exts = (0..inst_ext_count)
                .map(|i| std::ffi::CStr::from_ptr(*inst_exts.add(i as _).as_ref().unwrap()))
                .collect::<Vec<_>>();
            let device_exts = (0..device_ext_count)
                .map(|i| std::ffi::CStr::from_ptr(*device_exts.add(i as _).as_ref().unwrap()))
                .collect::<Vec<_>>();
            dbg!(inst_exts);
            dbg!(device_exts);*/

            let dlss_search_path = kajiya_backend::normalized_path_from_vfs("/kajiya").unwrap_or_else(|_| panic!("/kajiya VFS entry not found. Did you forget to call `set_standard_vfs_mount_points`?"));
            log::info!("DLSS DLL search path: {:?}", dlss_search_path);

            use std::os::windows::ffi::OsStrExt as _;
            let mut dlss_search_path_wchar = dlss_search_path
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0u16))
                .collect::<Vec<u16>>();

            let mut ngx_dll_paths = [dlss_search_path_wchar.as_mut_ptr()];
            let ngx_common_info = NVSDK_NGX_FeatureCommonInfo {
                PathListInfo: NVSDK_NGX_PathListInfo {
                    Path: ngx_dll_paths.as_mut_ptr(),
                    Length: ngx_dll_paths.len() as _,
                },
                InternalData: ptr::null_mut(),
                LoggingInfo: NGSDK_NGX_LoggingInfo {
                    LoggingCallback: None,
                    //MinimumLoggingLevel: NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_VERBOSE,
                    MinimumLoggingLevel: NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_OFF,
                    DisableOtherLoggingSinks: false,
                },
            };

            ngx_checked!(NVSDK_NGX_VULKAN_Init(
                0xcafebabe,
                wchz!(".").as_ptr(),
                transmute(backend.device.physical_device().instance.raw.handle()),
                transmute(backend.device.physical_device().raw),
                transmute(backend.device.raw.handle()),
                &ngx_common_info,
                NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            ));

            let mut ngx_params: *mut NVSDK_NGX_Parameter = ptr::null_mut();
            ngx_checked!(NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut ngx_params));

            let mut supersampling_needs_updated_driver = 0;
            ngx_checked!(NVSDK_NGX_Parameter_GetI(
                ngx_params,
                NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                &mut supersampling_needs_updated_driver
            ));
            assert_eq!(supersampling_needs_updated_driver, 0);

            let mut supersampling_available = 0;
            ngx_checked!(NVSDK_NGX_Parameter_GetI(
                ngx_params,
                NVSDK_NGX_Parameter_SuperSampling_Available,
                &mut supersampling_available
            ));
            assert_eq!(supersampling_available, 1);

            let quality_preference_order = [
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality,
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced,
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf,
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance,
            ];

            log::info!(
                "Finding a DLSS mode to produce {:?} output from {:?} input",
                target_resolution,
                input_resolution
            );

            let supported_quality_modes = quality_preference_order
                .iter()
                .copied()
                .filter_map(|quality_value| {
                    let settings = DlssOptimalSettings::for_target_resolution_at_quality(
                        ngx_params,
                        target_resolution,
                        quality_value,
                    );

                    if settings.supports_input_resolution(input_resolution) {
                        Some((quality_value, settings))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            let optimal_settings: Option<(NVSDK_NGX_PerfQuality_Value, DlssOptimalSettings)> =
                supported_quality_modes
                    .iter()
                    .find(|(_, settings)| {
                        input_resolution[0] >= settings.optimal_render_extent[0]
                            && input_resolution[1] >= settings.optimal_render_extent[1]
                    })
                    .copied()
                    .or_else(|| supported_quality_modes.first().copied());

            let (optimal_quality_value, optimal_settings) = if let Some(v) = optimal_settings {
                v
            } else {
                panic!(
                    "No DLSS quality mode can produce {:?} output from {:?} input",
                    target_resolution, input_resolution
                );
            };

            #[allow(non_upper_case_globals)]
            let quality_value_str = match optimal_quality_value {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf => "MaxPerf",
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced => "Balanced",
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality => "MaxQuality",
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance => {
                    "UltraPerformance"
                }
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraQuality => {
                    "UltraQuality"
                }
                _ => "unknown",
            };

            log::info!(
                "Using {} DLSS mode:\n{:#?}",
                quality_value_str,
                optimal_settings
            );

            let dlss_create_params = NVSDK_NGX_DLSS_Create_Params {
                Feature: NVSDK_NGX_Feature_Create_Params {
                    InWidth: optimal_settings.optimal_render_extent[0],
                    InHeight: optimal_settings.optimal_render_extent[1],
                    InTargetWidth: target_resolution[0],
                    InTargetHeight: target_resolution[1],
                    InPerfQualityValue: optimal_quality_value,
                },
                InFeatureCreateFlags:
                    NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR
                        | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
                        | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted,
                InEnableOutputSubrects: false,
            };
            //dbg!(&dlss_create_params);

            NVSDK_NGX_Parameter_SetUI(ngx_params, NVSDK_NGX_Parameter_CreationNodeMask, 1);
            NVSDK_NGX_Parameter_SetUI(ngx_params, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
            NVSDK_NGX_Parameter_SetUI(
                ngx_params,
                NVSDK_NGX_Parameter_Width,
                dlss_create_params.Feature.InWidth,
            );
            NVSDK_NGX_Parameter_SetUI(
                ngx_params,
                NVSDK_NGX_Parameter_Height,
                dlss_create_params.Feature.InHeight,
            );
            NVSDK_NGX_Parameter_SetUI(
                ngx_params,
                NVSDK_NGX_Parameter_OutWidth,
                dlss_create_params.Feature.InTargetWidth,
            );
            NVSDK_NGX_Parameter_SetUI(
                ngx_params,
                NVSDK_NGX_Parameter_OutHeight,
                dlss_create_params.Feature.InTargetHeight,
            );
            NVSDK_NGX_Parameter_SetI(
                ngx_params,
                NVSDK_NGX_Parameter_PerfQualityValue,
                dlss_create_params.Feature.InPerfQualityValue,
            );
            NVSDK_NGX_Parameter_SetI(
                ngx_params,
                NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                dlss_create_params.InFeatureCreateFlags,
            );
            NVSDK_NGX_Parameter_SetI(
                ngx_params,
                NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects,
                if dlss_create_params.InEnableOutputSubrects {
                    1
                } else {
                    0
                },
            );

            let mut dlss_feature: *mut NVSDK_NGX_Handle = ptr::null_mut();
            backend
                .device
                .with_setup_cb(|cb| {
                    ngx_checked!(NVSDK_NGX_VULKAN_CreateFeature(
                        transmute(cb),
                        NVSDK_NGX_Feature_NVSDK_NGX_Feature_SuperSampling,
                        ngx_params,
                        &mut dlss_feature,
                    ));
                })
                .map_err(|err| backend.device.report_error(err))
                .expect("NVSDK_NGX_VULKAN_CreateFeature (DLSS) failed");

            Self {
                dlss_feature,
                ngx_params,
                current_supersample_offset: Vec2::ZERO,
                frame_idx: 0,
            }
        }
    }

    pub fn render(
        &mut self,
        rg: &mut rg::TemporalRenderGraph,
        input: &rg::Handle<Image>,
        reprojection_map: &rg::Handle<Image>,
        depth: &rg::Handle<Image>,
        output_extent: [u32; 2],
    ) -> rg::Handle<Image> {
        let mut output = rg.create(
            ImageDesc::new_2d(vk::Format::R16G16B16A16_SFLOAT, output_extent).usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            ),
        );

        // Thanks to `InMVScaleX` and `InMVScaleY`, the reprojection map can be used directly.
        let motion_vectors = reprojection_map;

        let mut pass = rg.add_pass("dlss");
        let input_ref = pass.read(
            input,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let depth_ref = pass.read(
            depth,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let motion_vectors_ref = pass.read(
            motion_vectors,
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );
        let output_ref = pass.write(&mut output, AccessType::AnyShaderWrite);

        let input_extent = input.desc().extent_2d();
        let current_supersample_offset = self.current_supersample_offset;
        let dlss_feature = self.dlss_feature;
        let ngx_params = self.ngx_params;
        let should_reset = self.frame_idx == 0;

        pass.render(move |api| {
            let cb = api.cb;

            let mut input = image_to_ngx(api, input_ref, ImageViewDesc::default());
            let mut depth = image_to_ngx(
                api,
                depth_ref,
                ImageViewDesc {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    ..Default::default()
                },
            );
            let mut motion_vectors =
                image_to_ngx(api, motion_vectors_ref, ImageViewDesc::default());
            let mut output = image_to_ngx(api, output_ref, ImageViewDesc::default());

            let rendering_size = NVSDK_NGX_Dimensions {
                Width: input_extent[0],
                Height: input_extent[1],
            };
            let rendering_offset = NVSDK_NGX_Coordinates { X: 0, Y: 0 };

            let dlss_eval_params = NVSDK_NGX_VK_DLSS_Eval_Params {
                Feature: NVSDK_NGX_VK_Feature_Eval_Params {
                    pInColor: &mut input,
                    pInOutput: &mut output,
                    InSharpness: 0.0,
                },
                pInDepth: &mut depth,
                pInMotionVectors: &mut motion_vectors,
                InJitterOffsetX: -current_supersample_offset.x,
                InJitterOffsetY: current_supersample_offset.y,
                InRenderSubrectDimensions: rendering_size,
                InReset: if should_reset { 1 } else { 0 },
                InMVScaleX: input_extent[0] as f32,
                InMVScaleY: input_extent[1] as f32,
                pInTransparencyMask: ptr::null_mut(),
                pInExposureTexture: ptr::null_mut(),
                pInBiasCurrentColorMask: ptr::null_mut(),
                InColorSubrectBase: rendering_offset,
                InDepthSubrectBase: rendering_offset,
                InMVSubrectBase: rendering_offset,
                InTranslucencySubrectBase: rendering_offset,
                InBiasCurrentColorSubrectBase: rendering_offset,
                InOutputSubrectBase: rendering_offset,
                InPreExposure: 0.0,
                InIndicatorInvertXAxis: 0,
                InIndicatorInvertYAxis: 0,
                GBufferSurface: NVSDK_NGX_VK_GBuffer {
                    pInAttrib: [ptr::null_mut(); 16],
                },
                InToneMapperType: NVSDK_NGX_ToneMapperType_NVSDK_NGX_TONEMAPPER_STRING,
                pInMotionVectors3D: ptr::null_mut(),
                pInIsParticleMask: ptr::null_mut(),
                pInAnimatedTextureMask: ptr::null_mut(),
                pInDepthHighRes: ptr::null_mut(),
                pInPositionViewSpace: ptr::null_mut(),
                InFrameTimeDeltaInMsec: 0.0,
                pInRayTracingHitDistance: ptr::null_mut(),
                pInMotionVectorsReflections: ptr::null_mut(),
            };

            ngx_checked!(NGX_VULKAN_EVALUATE_DLSS_EXT(
                cb.raw,
                dlss_feature,
                ngx_params,
                dlss_eval_params
            ));

            Ok(())
        });

        self.frame_idx += 1;

        output
    }
}

#[derive(Debug, Clone, Copy)]
struct DlssOptimalSettings {
    optimal_render_extent: [u32; 2],
    max_render_extent: [u32; 2],
    min_render_extent: [u32; 2],
}

impl DlssOptimalSettings {
    fn for_target_resolution_at_quality(
        ngx_params: *mut NVSDK_NGX_Parameter,
        target_resolution: [u32; 2],
        value: NVSDK_NGX_PerfQuality_Value,
    ) -> Self {
        let mut optimal_render_extent = [0, 0];
        let mut max_render_extent = [0, 0];
        let mut min_render_extent = [0, 0];

        unsafe {
            let mut get_optimal_settings_fn = ptr::null_mut();
            ngx_checked!(NVSDK_NGX_Parameter_GetVoidPointer(
                ngx_params,
                NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback,
                &mut get_optimal_settings_fn,
            ));
            assert_ne!(get_optimal_settings_fn, ptr::null_mut());

            let get_optimal_settings_fn: PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback =
                transmute(get_optimal_settings_fn);

            NVSDK_NGX_Parameter_SetUI(ngx_params, NVSDK_NGX_Parameter_Width, target_resolution[0]);
            NVSDK_NGX_Parameter_SetUI(ngx_params, NVSDK_NGX_Parameter_Height, target_resolution[1]);
            NVSDK_NGX_Parameter_SetI(ngx_params, NVSDK_NGX_Parameter_PerfQualityValue, value);
            NVSDK_NGX_Parameter_SetI(ngx_params, NVSDK_NGX_Parameter_RTXValue, 0); // Some older DLSS dlls still expect this value to be set
            ngx_checked!(get_optimal_settings_fn(ngx_params));

            NVSDK_NGX_Parameter_GetUI(
                ngx_params,
                NVSDK_NGX_Parameter_OutWidth,
                &mut optimal_render_extent[0],
            );
            NVSDK_NGX_Parameter_GetUI(
                ngx_params,
                NVSDK_NGX_Parameter_OutHeight,
                &mut optimal_render_extent[1],
            );
            NVSDK_NGX_Parameter_GetUI(
                ngx_params,
                NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Width,
                &mut max_render_extent[0],
            );
            NVSDK_NGX_Parameter_GetUI(
                ngx_params,
                NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Height,
                &mut max_render_extent[1],
            );
            NVSDK_NGX_Parameter_GetUI(
                ngx_params,
                NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Width,
                &mut min_render_extent[0],
            );
            NVSDK_NGX_Parameter_GetUI(
                ngx_params,
                NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Height,
                &mut min_render_extent[1],
            );
        }

        Self {
            optimal_render_extent,
            max_render_extent,
            min_render_extent,
        }
    }

    fn supports_input_resolution(&self, input_resolution: [u32; 2]) -> bool {
        input_resolution[0] >= self.min_render_extent[0]
            && input_resolution[1] >= self.min_render_extent[1]
            && input_resolution[0] <= self.max_render_extent[0]
            && input_resolution[1] <= self.max_render_extent[1]
    }
}

#[allow(non_snake_case)]
fn NGX_VULKAN_EVALUATE_DLSS_EXT(
    InCmdList: vk::CommandBuffer,
    pInHandle: *mut NVSDK_NGX_Handle,
    pInParams: *mut NVSDK_NGX_Parameter,
    pInDlssEvalParams: NVSDK_NGX_VK_DLSS_Eval_Params,
) -> NVSDK_NGX_Result {
    unsafe {
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_Color,
            pInDlssEvalParams.Feature.pInColor as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_Output,
            pInDlssEvalParams.Feature.pInOutput as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_Depth,
            pInDlssEvalParams.pInDepth as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_MotionVectors,
            pInDlssEvalParams.pInMotionVectors as _,
        );
        NVSDK_NGX_Parameter_SetF(
            pInParams,
            NVSDK_NGX_Parameter_Jitter_Offset_X,
            pInDlssEvalParams.InJitterOffsetX,
        );
        NVSDK_NGX_Parameter_SetF(
            pInParams,
            NVSDK_NGX_Parameter_Jitter_Offset_Y,
            pInDlssEvalParams.InJitterOffsetY,
        );
        NVSDK_NGX_Parameter_SetF(
            pInParams,
            NVSDK_NGX_Parameter_Sharpness,
            pInDlssEvalParams.Feature.InSharpness,
        );
        NVSDK_NGX_Parameter_SetI(
            pInParams,
            NVSDK_NGX_Parameter_Reset,
            pInDlssEvalParams.InReset,
        );
        NVSDK_NGX_Parameter_SetF(
            pInParams,
            NVSDK_NGX_Parameter_MV_Scale_X,
            if pInDlssEvalParams.InMVScaleX == 0.0 {
                1.0
            } else {
                pInDlssEvalParams.InMVScaleX
            },
        );
        NVSDK_NGX_Parameter_SetF(
            pInParams,
            NVSDK_NGX_Parameter_MV_Scale_Y,
            if pInDlssEvalParams.InMVScaleY == 0.0 {
                1.0
            } else {
                pInDlssEvalParams.InMVScaleY
            },
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_TransparencyMask,
            pInDlssEvalParams.pInTransparencyMask as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_ExposureTexture,
            pInDlssEvalParams.pInExposureTexture as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask,
            pInDlssEvalParams.pInBiasCurrentColorMask as _,
        );
        /*NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Albedo, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_ALBEDO]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Roughness, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_ROUGHNESS]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Metallic, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_METALLIC]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Specular, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_SPECULAR]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Subsurface, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_SUBSURFACE]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Normals, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_NORMALS]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_ShadingModelId, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_SHADINGMODELID]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_MaterialId, pInDlssEvalParams.GBufferSurface.pInAttrib[NVSDK_NGX_GBUFFER_MATERIALID]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_8, pInDlssEvalParams.GBufferSurface.pInAttrib[8]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_9, pInDlssEvalParams.GBufferSurface.pInAttrib[9]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_10, pInDlssEvalParams.GBufferSurface.pInAttrib[10]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_11, pInDlssEvalParams.GBufferSurface.pInAttrib[11]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_12, pInDlssEvalParams.GBufferSurface.pInAttrib[12]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_13, pInDlssEvalParams.GBufferSurface.pInAttrib[13]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_14, pInDlssEvalParams.GBufferSurface.pInAttrib[14]);
        NVSDK_NGX_Parameter_SetVoidPointer(pInParams, NVSDK_NGX_Parameter_GBuffer_Atrrib_15, pInDlssEvalParams.GBufferSurface.pInAttrib[15]);*/
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_TonemapperType,
            pInDlssEvalParams.InToneMapperType as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_MotionVectors3D,
            pInDlssEvalParams.pInMotionVectors3D as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_IsParticleMask,
            pInDlssEvalParams.pInIsParticleMask as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_AnimatedTextureMask,
            pInDlssEvalParams.pInAnimatedTextureMask as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_DepthHighRes,
            pInDlssEvalParams.pInDepthHighRes as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_Position_ViewSpace,
            pInDlssEvalParams.pInPositionViewSpace as _,
        );
        NVSDK_NGX_Parameter_SetF(
            pInParams,
            NVSDK_NGX_Parameter_FrameTimeDeltaInMsec,
            pInDlssEvalParams.InFrameTimeDeltaInMsec,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_RayTracingHitDistance,
            pInDlssEvalParams.pInRayTracingHitDistance as _,
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            pInParams,
            NVSDK_NGX_Parameter_MotionVectorsReflection,
            pInDlssEvalParams.pInMotionVectorsReflections as _,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,
            pInDlssEvalParams.InColorSubrectBase.X,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y,
            pInDlssEvalParams.InColorSubrectBase.Y,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X,
            pInDlssEvalParams.InDepthSubrectBase.X,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y,
            pInDlssEvalParams.InDepthSubrectBase.Y,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,
            pInDlssEvalParams.InMVSubrectBase.X,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y,
            pInDlssEvalParams.InMVSubrectBase.Y,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_X,
            pInDlssEvalParams.InTranslucencySubrectBase.X,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_Y,
            pInDlssEvalParams.InTranslucencySubrectBase.Y,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_X,
            pInDlssEvalParams.InBiasCurrentColorSubrectBase.X,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_Y,
            pInDlssEvalParams.InBiasCurrentColorSubrectBase.Y,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X,
            pInDlssEvalParams.InOutputSubrectBase.X,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y,
            pInDlssEvalParams.InOutputSubrectBase.Y,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
            pInDlssEvalParams.InRenderSubrectDimensions.Width,
        );
        NVSDK_NGX_Parameter_SetUI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
            pInDlssEvalParams.InRenderSubrectDimensions.Height,
        );
        NVSDK_NGX_Parameter_SetF(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Pre_Exposure,
            if pInDlssEvalParams.InPreExposure == 0.0 {
                1.0
            } else {
                pInDlssEvalParams.InPreExposure
            },
        );
        NVSDK_NGX_Parameter_SetI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis,
            pInDlssEvalParams.InIndicatorInvertXAxis,
        );
        NVSDK_NGX_Parameter_SetI(
            pInParams,
            NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis,
            pInDlssEvalParams.InIndicatorInvertYAxis,
        );

        NVSDK_NGX_VULKAN_EvaluateFeature_C(transmute(InCmdList), pInHandle, pInParams, None)
    }
}

fn image_to_ngx<ViewType: rg::GpuViewType>(
    api: &rg::RenderPassApi,
    image_ref: rg::Ref<Image, ViewType>,
    view_desc: ImageViewDesc,
) -> NVSDK_NGX_Resource_VK {
    let device = api.device();
    let image = api.resources.image(image_ref);

    let view = image.view(device, &view_desc).unwrap();
    let view_desc = image.view_desc(&view_desc);

    unsafe {
        NVSDK_NGX_Resource_VK {
            Resource: NVSDK_NGX_Resource_VK__bindgen_ty_1 {
                ImageViewInfo: NVSDK_NGX_ImageViewInfo_VK {
                    ImageView: transmute(view),
                    Image: transmute(image.raw),
                    SubresourceRange: transmute(view_desc.subresource_range),
                    Format: transmute(view_desc.format),
                    Width: image.desc.extent[0],
                    Height: image.desc.extent[1],
                },
            }, // NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW
            Type: NVSDK_NGX_Resource_VK_Type_NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
            ReadWrite: ViewType::IS_WRITABLE,
        }
    }
}

mod ngx_params {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]

    pub const NVSDK_NGX_Parameter_SuperSampling_Available: *const i8 =
        "SuperSampling.Available\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver: *const i8 =
        "SuperSampling.NeedsUpdatedDriver\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Width: *const i8 = "Width\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Height: *const i8 = "Height\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_OutWidth: *const i8 = "OutWidth\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_OutHeight: *const i8 = "OutHeight\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Sharpness: *const i8 = "Sharpness\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Color: *const i8 = "Color\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Output: *const i8 = "Output\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Reset: *const i8 = "Reset\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_MotionVectors: *const i8 =
        "MotionVectors\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_MV_Scale_X: *const i8 = "MV.Scale.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_MV_Scale_Y: *const i8 = "MV.Scale.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_CreationNodeMask: *const i8 =
        "CreationNodeMask\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_VisibilityNodeMask: *const i8 =
        "VisibilityNodeMask\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Depth: *const i8 = "Depth\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback: *const i8 =
        "DLSSOptimalSettingsCallback\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_PerfQualityValue: *const i8 =
        "PerfQualityValue\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_RTXValue: *const i8 = "RTXValue\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Jitter_Offset_X: *const i8 =
        "Jitter.Offset.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Jitter_Offset_Y: *const i8 =
        "Jitter.Offset.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_TransparencyMask: *const i8 =
        "TransparencyMask\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_ExposureTexture: *const i8 =
        "ExposureTexture\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags: *const i8 =
        "DLSS.Feature.Create.Flags\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_TonemapperType: *const i8 =
        "TonemapperType\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_MotionVectors3D: *const i8 =
        "MotionVectors3D\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_IsParticleMask: *const i8 =
        "IsParticleMask\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_AnimatedTextureMask: *const i8 =
        "AnimatedTextureMask\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DepthHighRes: *const i8 = "DepthHighRes\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_Position_ViewSpace: *const i8 =
        "Position.ViewSpace\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_FrameTimeDeltaInMsec: *const i8 =
        "FrameTimeDeltaInMsec\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_RayTracingHitDistance: *const i8 =
        "RayTracingHitDistance\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_MotionVectorsReflection: *const i8 =
        "MotionVectorsReflection\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects: *const i8 =
        "DLSS.Enable.Output.Subrects\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X: *const i8 =
        "DLSS.Input.Color.Subrect.Base.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y: *const i8 =
        "DLSS.Input.Color.Subrect.Base.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X: *const i8 =
        "DLSS.Input.Depth.Subrect.Base.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y: *const i8 =
        "DLSS.Input.Depth.Subrect.Base.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X: *const i8 =
        "DLSS.Input.MV.Subrect.Base.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y: *const i8 =
        "DLSS.Input.MV.Subrect.Base.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_X: *const i8 =
        "DLSS.Input.Translucency.Subrect.Base.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Translucency_SubrectBase_Y: *const i8 =
        "DLSS.Input.Translucency.Subrect.Base.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X: *const i8 =
        "DLSS.Output.Subrect.Base.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y: *const i8 =
        "DLSS.Output.Subrect.Base.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width: *const i8 =
        "DLSS.Render.Subrect.Dimensions.Width\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height: *const i8 =
        "DLSS.Render.Subrect.Dimensions.Height\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Pre_Exposure: *const i8 =
        "DLSS.Pre.Exposure\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Exposure_Scale: *const i8 =
        "DLSS.Exposure.Scale\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask: *const i8 =
        "DLSS.Input.Bias.Current.Color.Mask\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_X: *const i8 =
        "DLSS.Input.Bias.Current.Color.Subrect.Base.X\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_Y: *const i8 =
        "DLSS.Input.Bias.Current.Color.Subrect.Base.Y\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis: *const i8 =
        "DLSS.Indicator.Invert.Y.Axis\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis: *const i8 =
        "DLSS.Indicator.Invert.X.Axis\0".as_ptr() as *const i8;

    pub const NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Width: *const i8 =
        "DLSS.Get.Dynamic.Max.Render.Width\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Max_Render_Height: *const i8 =
        "DLSS.Get.Dynamic.Max.Render.Height\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Width: *const i8 =
        "DLSS.Get.Dynamic.Min.Render.Width\0".as_ptr() as *const i8;
    pub const NVSDK_NGX_Parameter_DLSS_Get_Dynamic_Min_Render_Height: *const i8 =
        "DLSS.Get.Dynamic.Min.Render.Height\0".as_ptr() as *const i8;
}
use ngx_params::*;

#[allow(non_camel_case_types)]
type PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback =
    extern "cdecl" fn(*mut NVSDK_NGX_Parameter) -> NVSDK_NGX_Result;
