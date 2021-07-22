#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/image.hlsl"
#include "../inc/soft_color_clamp.hlsl"

[[vk::binding(0)]] Texture2D<float4> shadow_mask_tex;
[[vk::binding(1)]] Texture2D<uint> bitpacked_shadow_mask_tex;
[[vk::binding(2)]] Texture2D<float4> prev_moments_tex;
[[vk::binding(3)]] Texture2D<float4> prev_accum_tex;
[[vk::binding(4)]] Texture2D<float4> reprojection_tex;
[[vk::binding(5)]] RWTexture2D<float4> output_moments_tex;
[[vk::binding(6)]] RWTexture2D<float2> temporal_output_tex;
[[vk::binding(7)]] RWTexture2D<uint> meta_output_tex;
[[vk::binding(8)]] cbuffer _ {
    float4 input_tex_size;
    uint2 bitpacked_shadow_mask_extent;
};

uint2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return uint2(input_tex_size.xy);
}

float2 FFX_DNSR_Shadows_GetInvBufferDimensions() {
    return input_tex_size.zw;
}

float4x4 FFX_DNSR_Shadows_GetProjectionInverse() {
    return frame_constants.view_constants.clip_to_view;
}

float4x4 FFX_DNSR_Shadows_GetReprojectionMatrix() {
    return frame_constants.view_constants.clip_to_prev_clip;
}

float4x4 FFX_DNSR_Shadows_GetViewProjectionInverse() {
    // TODO: replace the temporal component in the denoiser
    return frame_constants.view_constants.clip_to_view;
}

float3 FFX_DNSR_Shadows_GetEye() {
    return get_eye_position();
}

float3 FFX_DNSR_Shadows_ReadNormals(uint2 px) {
    // TODO
    return float3(0, 0, 1);
}

float FFX_DNSR_Shadows_ReadDepth(uint2 px) {
    // TODO
    return 0.5;
}

float FFX_DNSR_Shadows_ReadPreviousDepth(uint2 px) {
    // TODO
    return 0.5;
}

bool FFX_DNSR_Shadows_IsShadowReciever(uint2 px) {
    // TODO
    return true;
}

float2 FFX_DNSR_Shadows_ReadVelocity(uint2 px) {
    // TODO
    return 0.0.xx;
}

void FFX_DNSR_Shadows_WriteMetadata(uint linear_tile_index, uint mask) {
    const uint2 tile = uint2(
        linear_tile_index % bitpacked_shadow_mask_extent.x,
        linear_tile_index / bitpacked_shadow_mask_extent.x
    );

    meta_output_tex[tile] = mask;
}

uint FFX_DNSR_Shadows_ReadRaytracedShadowMask(uint linear_tile_index) {
    const uint2 tile = uint2(
        linear_tile_index % bitpacked_shadow_mask_extent.x,
        linear_tile_index / bitpacked_shadow_mask_extent.x
    );

    return bitpacked_shadow_mask_tex[tile];
}

void FFX_DNSR_Shadows_WriteReprojectionResults(uint2 px, float2 shadow_clamped_variance) {
    temporal_output_tex[px] = shadow_clamped_variance;
}

void FFX_DNSR_Shadows_WriteMoments(uint2 px, float4 moments) {
    // Don't accumulate more samples than a certain number,
    // so that our variance estimate is quick, and contact shadows turn crispy sooner.
    moments.z = min(moments.z, 32);

    output_moments_tex[px] = moments;
}

float FFX_DNSR_Shadows_HitsLight(uint2 px) {
    return shadow_mask_tex[px].x;
}

struct HistoryRemap {
    static HistoryRemap create() {
        HistoryRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return v;
    }
};

float4 FFX_DNSR_Shadows_ReadPreviousMomentsBuffer(float2 uv) {
    #if 1
        float4 moments = image_sample_catmull_rom(
            TextureImage::from_parts(prev_moments_tex, input_tex_size.xy),
            uv,
            HistoryRemap::create()
        );
        // Clamp EX2 and sample count
        moments.yz = max(0, moments.yz);
        return moments;
    #else
        return prev_moments_tex.SampleLevel(sampler_lnc, uv, 0);
    #endif
}

float FFX_DNSR_Shadows_ReadHistory(float2 uv) {
    #if 1
        return image_sample_catmull_rom(
            TextureImage::from_parts(prev_accum_tex, input_tex_size.xy),
            uv,
            HistoryRemap::create()
        ).x;
    #else
        return prev_accum_tex.SampleLevel(sampler_lnc, uv, 0).x;
    #endif
}

bool FFX_DNSR_Shadows_IsFirstFrame() {
    return frame_constants.frame_index == 0;
}

#include "ffx/ffx_denoiser_shadows_tileclassification.hlsl"

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID, uint group_index: SV_GroupIndex, uint2 gid: SV_GroupID) {
    FFX_DNSR_Shadows_TileClassification(group_index, gid);
}
