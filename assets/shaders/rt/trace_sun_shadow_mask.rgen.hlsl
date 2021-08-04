#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/bindless_textures.hlsl"

#include "../inc/blue_noise.hlsl"
#include "../inc/math.hlsl"

#define USE_SOFT_SHADOWS 1

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0)]] Texture2D<float> depth_tex;
[[vk::binding(1)]] Texture2D<float3> geometric_normal_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;

    const float2 pixel_center = px + 0.5.xx;
    const float2 uv = pixel_center / DispatchRaysDimensions().xy;

    float z_over_w = depth_tex[px];
    if (0.0 == z_over_w) {
        output_tex[px] = 1.0;
        return;
    }

    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_vs = mul(frame_constants.view_constants.sample_to_view, pt_cs);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, pt_vs);
    pt_ws /= pt_ws.w;
    pt_vs /= pt_vs.w;

    float4 eye_ws = mul(frame_constants.view_constants.view_to_world, float4(0, 0, 0, 1));

    const float3 normal_vs = geometric_normal_tex[px] * 2.0 - 1.0;
    const float3 normal_ws = mul(frame_constants.view_constants.view_to_world, float4(normal_vs, 0.0)).xyz;

    const float3 bias_dir = normal_ws;
    const float bias_amount = (-pt_vs.z + length(pt_ws.xyz)) * 1e-5;
    const float3 ray_origin = pt_ws.xyz + bias_dir * bias_amount;

    const bool is_shadowed = rt_is_shadowed(
        acceleration_structure,
        new_ray(
            ray_origin,
            sample_sun_direction(
                blue_noise_for_pixel(px, frame_constants.frame_index).xy,
                USE_SOFT_SHADOWS
            ),
            0,
            FLT_MAX
        ));

    output_tex[px] = is_shadowed ? 0.0 : 1.0;
}
