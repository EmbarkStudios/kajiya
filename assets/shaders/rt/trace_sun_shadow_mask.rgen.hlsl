#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"

#include "../inc/hash.hlsl"
#include "../inc/math.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] Texture2D<float> depth_tex;
[[vk::binding(1, 0)]] RWTexture2D<float4> output_tex;

float3 sample_sun_direction(uint2 px) {
    const float3x3 basis = build_orthonormal_basis(normalize(SUN_DIRECTION));

    uint rng = hash3(uint3(px, frame_constants.frame_index));
    float2 urand = float2(
        uint_to_u01_float(hash1_mut(rng)),
        uint_to_u01_float(hash1_mut(rng))
    );

    return mul(basis, uniform_sample_cone(urand, 0.998));
}

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

    const float3 bias_dir = normalize(eye_ws.xyz / eye_ws.w - pt_ws.xyz);

    const float3 ray_origin = pt_ws.xyz + bias_dir * (length(pt_vs.xyz) + 10 * length(pt_ws.xyz)) * 1e-4;
    const bool is_shadowed = rt_is_shadowed(
        acceleration_structure,
        new_ray(
            ray_origin,
            sample_sun_direction(px),
            0,
            FLT_MAX
        ));

    output_tex[px] = is_shadowed ? 0.0 : 1.0;
}
