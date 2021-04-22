#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<uint> bitpacked_shadow_mask_tex;
[[vk::binding(1)]] Texture2D<float3> prev_moments_tex;
[[vk::binding(2)]] Texture2D<float> prev_accum_tex;
[[vk::binding(3)]] Texture2D<float4> reprojection_tex;
[[vk::binding(4)]] RWTexture2D<float3> output_moments_tex;
[[vk::binding(5)]] RWTexture2D<float2> temporal_output_tex;
[[vk::binding(6)]] RWTexture2D<uint> meta_output_tex;
[[vk::binding(7)]] cbuffer _ {
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

void FFX_DNSR_Shadows_WriteMoments(uint2 px, float3 moments) {
    //moments.z = min(moments.z, 32);
    output_moments_tex[px] = moments;
}

float3 FFX_DNSR_Shadows_ReadPreviousMomentsBuffer(uint2 px) {
    return prev_moments_tex[px];
}

float FFX_DNSR_Shadows_ReadHistory(float2 uv) {
    return prev_accum_tex.SampleLevel(sampler_lnc, uv, 0);
}

bool FFX_DNSR_Shadows_IsFirstFrame() {
    return frame_constants.frame_index == 0;
}

#include "ffx/ffx_denoiser_shadows_tileclassification.hlsl"

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID, uint group_index: SV_GroupIndex, uint2 gid: SV_GroupID) {
    FFX_DNSR_Shadows_TileClassification(group_index, gid);
}
