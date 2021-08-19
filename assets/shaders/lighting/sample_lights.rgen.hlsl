#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/lights/triangle.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float> depth_tex;
[[vk::binding(1)]] RWTexture2D<float4> out0_tex;
[[vk::binding(2)]] RWTexture2D<float4> out1_tex;
[[vk::binding(3)]] RWTexture2D<float4> out2_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };

    const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        out0_tex[px] = 0.0.xxxx;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);
    const float3 shadow_ray_origin = view_ray_context.biased_secondary_ray_origin_ws();

    const float3 urand3 = blue_noise_for_pixel(px, frame_constants.frame_index).xyz;
    const float2 urand = urand3.xy;

    const uint light_count = frame_constants.triangle_light_count;
    const uint light_idx = uint(urand3.z * light_count) % light_count;
    //uint rng = hash3(uint3(px, frame_constants.frame_index));
    //const uint light_idx = rng % light_count;
    const float light_choice_pmf = 1.0 / light_count;

    TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
    LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
    const float3 to_light_ws = light_sample.pos - shadow_ray_origin;
    const float dist_to_light = length(to_light_ws);

    const bool is_shadowed =
        rt_is_shadowed(
            acceleration_structure,
            new_ray(
                shadow_ray_origin,
                to_light_ws / max(1e-8, dist_to_light),
                0,
                dist_to_light - 1e-4
        ));

    out0_tex[px] = float4(is_shadowed ? 0 : triangle_light.radiance(), 1);
    out1_tex[px] = float4(
        view_ray_context.ray_hit_vs() + direction_world_to_view(to_light_ws),
        light_sample.pdf.value * light_choice_pmf
    );
    out2_tex[px] = float4(direction_world_to_view(light_sample.normal), 0);
}
