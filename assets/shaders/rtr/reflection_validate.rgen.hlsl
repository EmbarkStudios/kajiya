#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../inc/reservoir.hlsl"
#include "../ircache/bindings.hlsl"
#include "../wrc/bindings.hlsl"
#include "rtr_settings.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> rtdgi_tex;
[[vk::binding(3)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(4)]] RWTexture2D<float> refl_restir_invalidity_tex;
DEFINE_IRCACHE_BINDINGS(5, 6, 7, 8, 9, 10, 11, 12, 13)
DEFINE_WRC_BINDINGS(14)
[[vk::binding(15)]] Texture2D<float4> ray_orig_history_tex;
[[vk::binding(16)]] Texture2D<float4> ray_history_tex;
[[vk::binding(17)]] Texture2D<uint> rng_history_tex;
[[vk::binding(18)]] RWTexture2D<float4> irradiance_history_tex;
[[vk::binding(19)]] RWTexture2D<uint2> reservoir_history_tex;
[[vk::binding(20)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

//#define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125
#include "../ircache/lookup.hlsl"
#include "../wrc/lookup.hlsl"

#include "reflection_trace_common.inc.hlsl"

[shader("raygeneration")]
void main() {
    if (!RTR_RESTIR_USE_PATH_VALIDATION) {
        return;
    }

    // Validation at half-res
    const uint2 px = DispatchRaysIndex().xy * 2 + HALFRES_SUBSAMPLE_OFFSET;
    //const uint2 px = DispatchRaysIndex().xy;

    // Standard jitter from the other reflection passes
    const uint2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;
    float depth = depth_tex[hi_px];

    //refl_restir_invalidity_tex[px] = 0;

    if (0.0 == depth) {
        refl_restir_invalidity_tex[px] = 1;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    gbuffer.roughness = max(gbuffer.roughness, RTR_ROUGHNESS_CLAMP);

    const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_biased_depth(uv, depth);
    const float3 refl_ray_origin_ws = view_ray_context.biased_secondary_ray_origin_ws_with_normal(gbuffer.normal);
#else
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);
    const float3 refl_ray_origin_ws = view_ray_context.biased_secondary_ray_origin_ws();
#endif

    // TODO: frame consistency
    const uint noise_offset = frame_constants.frame_index * select(USE_TEMPORAL_JITTER, 1, 0);

    const float3 ray_orig_ws = ray_orig_history_tex[px].xyz + get_prev_eye_position();
    const float3 ray_hit_ws = ray_history_tex[px].xyz + ray_orig_ws;

    RayDesc outgoing_ray;
    outgoing_ray.Direction = normalize(ray_hit_ws - ray_orig_ws);
    outgoing_ray.Origin = ray_orig_ws;
    outgoing_ray.TMin = 0;
    outgoing_ray.TMax = SKY_DIST;

    //uint rng = hash2(px);
    uint rng = rng_history_tex[px];
    RtrTraceResult result = do_the_thing(px, gbuffer.normal, gbuffer.roughness, rng, outgoing_ray);

    Reservoir1spp r = Reservoir1spp::from_raw(reservoir_history_tex[px]);

    const float4 prev_irradiance_packed = irradiance_history_tex[px];
    const float3 prev_irradiance = max(0.0.xxx, prev_irradiance_packed.rgb * frame_constants.pre_exposure_delta);
    const float3 check_radiance = max(0.0.xxx, result.total_radiance);

    const float rad_diff = length(abs(prev_irradiance - check_radiance) / max(1e-3, prev_irradiance + check_radiance));
    const float invalidity = smoothstep(0.1, 0.5, rad_diff / length(1.0.xxx));

    //r.M = max(0, min(r.M, exp2(log2(float(RTR_RESTIR_TEMPORAL_M_CLAMP)) * (1.0 - invalidity))));
    r.M *= 1 - invalidity;

    // TODO: also update hit point and normal
    // TODO: does this also need a hit_t check as in rtdgi restir validation?
    // TOD:: rename to radiance
    irradiance_history_tex[px] = float4(check_radiance, prev_irradiance_packed.a);

    refl_restir_invalidity_tex[px] = invalidity;
    reservoir_history_tex[px] = r.as_raw();

    // Also reduce M of the neighbors in case we have fewer validation rays than irradiance rays.
    #if 1
        for (uint i = 1; i <= 3; ++i) {
            //const uint2 main_px = px;
            //const uint2 px = (main_px & ~1u) + HALFRES_SUBSAMPLE_OFFSET;
            const uint2 px = DispatchRaysIndex().xy * 2 + hi_px_subpixels[(frame_constants.frame_index + i) & 3];

            const float4 neighbor_prev_irradiance_packed = irradiance_history_tex[px];
            {
                const float3 a = max(0.0.xxx, neighbor_prev_irradiance_packed.rgb * frame_constants.pre_exposure_delta);
                const float3 b = prev_irradiance;
                const float neigh_rad_diff = length(abs(a - b) / max(1e-8, a + b));

                // If the neighbor and us tracked similar radiance, assume it would also have
                // a similar change in value upon validation.
                if (neigh_rad_diff < 0.2) {
                    // With this assumption, we'll replace the neighbor's old radiance with our own new one.
                    irradiance_history_tex[px] = float4(check_radiance, neighbor_prev_irradiance_packed.a);
                }
            }

            refl_restir_invalidity_tex[px] = invalidity;

            if (invalidity > 0) {
                Reservoir1spp r = Reservoir1spp::from_raw(reservoir_history_tex[px]);
                //r.M = max(0, min(r.M, exp2(log2(float(RTR_RESTIR_TEMPORAL_M_CLAMP)) * (1.0 - invalidity))));
                r.M *= 1 - invalidity;
                reservoir_history_tex[px] = r.as_raw();
            }
        }
    #endif
}
