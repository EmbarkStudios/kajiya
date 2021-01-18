#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/rt.hlsl"

static const float3 SUN_DIRECTION = normalize(float3(1, 0.3, 1));

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] Texture2D<float> depth_tex;
[[vk::binding(1, 0)]] RWTexture2D<float4> output_tex;

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;

    const float2 pixel_center = px + 0.5.xx;
    const float2 uv = pixel_center / DispatchRaysDimensions().xy;

    float z_over_w = depth_tex[px];
    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_vs = mul(frame_constants.view_constants.sample_to_view, pt_cs);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, pt_vs);
    pt_ws /= pt_ws.w;

    const bool is_shadowed = rt_is_shadowed(
        acceleration_structure,
        new_ray(
            pt_ws.xyz,
            SUN_DIRECTION,
            -pt_vs.z / pt_vs.w * 1e-3 + 1e-4,
            FLT_MAX
        ));

    output_tex[px] = is_shadowed ? 0.0 : 1.0;
}
