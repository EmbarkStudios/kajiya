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
#include "../rtr/rtr_settings.hlsl" // for rtr_encode_cos_theta_for_fp16. consider moving out.
#include "rtdgi_restir_settings.hlsl"
#include "near_field_settings.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float3> half_view_normal_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> reprojected_gi_tex;
[[vk::binding(3)]] Texture2D<float4> reprojection_tex;
DEFINE_IRCACHE_BINDINGS(4, 5, 6, 7, 8, 9, 10, 11, 12)
DEFINE_WRC_BINDINGS(13)
[[vk::binding(14)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(15)]] Texture2D<float3> ray_orig_history_tex;
[[vk::binding(16)]] RWTexture2D<float4> candidate_irradiance_out_tex;
[[vk::binding(17)]] RWTexture2D<float4> candidate_normal_out_tex;
[[vk::binding(18)]] RWTexture2D<float4> candidate_hit_out_tex;
[[vk::binding(19)]] Texture2D<float> rt_history_invalidity_in_tex;
[[vk::binding(20)]] RWTexture2D<float> rt_history_invalidity_out_tex;
[[vk::binding(21)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

//#define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
//#define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125

#include "../ircache/lookup.hlsl"
#include "../wrc/lookup.hlsl"
#include "candidate_ray_dir.hlsl"

#include "diffuse_trace_common.inc.hlsl"


[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    const int2 hi_px_offset = HALFRES_SUBSAMPLE_OFFSET;
    const uint2 hi_px = px * 2 + hi_px_offset;
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        candidate_irradiance_out_tex[px] = 0;
        candidate_normal_out_tex[px] = float4(0, 0, 1, 0);
        rt_history_invalidity_out_tex[px] = 0;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_biased_depth(uv, depth);

    const float NEAR_FIELD_FADE_OUT_END = -view_ray_context.ray_hit_vs().z * (SSGI_NEAR_FIELD_RADIUS * gbuffer_tex_size.w * 0.5);

    #if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
        if (true) {
    #else
        if (is_rtdgi_tracing_frame()) {
    #endif
        const float3 normal_vs = half_view_normal_tex[px];
        const float3 normal_ws = direction_view_to_world(normal_vs);
        const float3x3 tangent_to_world = build_orthonormal_basis(normal_ws);
        const float3 outgoing_dir = rtdgi_candidate_ray_dir(px, tangent_to_world);

        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = view_ray_context.biased_secondary_ray_origin_ws_with_normal(normal_ws);
        outgoing_ray.TMin = 0;

        if (is_rtdgi_tracing_frame()) {
            outgoing_ray.TMax = SKY_DIST;
        } else {
            outgoing_ray.TMax = NEAR_FIELD_FADE_OUT_END;
        }

        uint rng = hash3(uint3(px, frame_constants.frame_index & 31));
        TraceResult result = do_the_thing(px, normal_ws, rng, outgoing_ray);

        #if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
            if (!is_rtdgi_tracing_frame() && !result.is_hit) {
                // If we were only tracing short rays, make sure we don't try to output
                // sky color upon misses.
                result.out_value = 0;
                result.hit_t = SKY_DIST;
            }
        #endif

        const float3 hit_offset_ws = outgoing_ray.Direction * result.hit_t;

        const float cos_theta = dot(normalize(outgoing_dir - view_ray_context.ray_dir_ws()), normal_ws);
        candidate_irradiance_out_tex[px] = float4(result.out_value, rtr_encode_cos_theta_for_fp16(cos_theta));
        candidate_hit_out_tex[px] = float4(hit_offset_ws, result.pdf * select(is_rtdgi_tracing_frame(), 1, -1));
        candidate_normal_out_tex[px] = float4(direction_world_to_view(result.hit_normal_ws), 0);
    } else {
        const float4 reproj = reprojection_tex[hi_px];
        const int2 reproj_px = floor(px + gbuffer_tex_size.xy * reproj.xy / 2 + 0.5);

        candidate_irradiance_out_tex[px] = 0.0;
        candidate_hit_out_tex[px] = 0.0;
        candidate_normal_out_tex[px] = 0.0;
    }

    const float4 reproj = reprojection_tex[hi_px];
    const int2 reproj_px = floor(px + gbuffer_tex_size.xy * reproj.xy / 2 + 0.5);
    rt_history_invalidity_out_tex[px] = rt_history_invalidity_in_tex[reproj_px];
}
