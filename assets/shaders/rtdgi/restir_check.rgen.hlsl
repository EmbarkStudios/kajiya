#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/reservoir.hlsl"
#include "rtdgi_restir_settings.hlsl"
#include "rtdgi_common.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float> half_depth_tex;
[[vk::binding(1)]] Texture2D<uint4> temporal_reservoir_packed_tex;
[[vk::binding(2)]] RWTexture2D<uint2> reservoir_input_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    const int2 hi_px_offset = HALFRES_SUBSAMPLE_OFFSET;
    const uint2 hi_px = px * 2 + hi_px_offset;

    const float depth = half_depth_tex[px];
    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_biased_depth(uv, depth);

    Reservoir1spp r = Reservoir1spp::from_raw(reservoir_input_tex[px]);
    const uint2 spx = reservoir_payload_to_px(r.payload);
    const TemporalReservoirOutput spx_packed = TemporalReservoirOutput::from_raw(temporal_reservoir_packed_tex[spx]);

    const float2 spx_uv = get_uv(
        spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
        gbuffer_tex_size);
    const ViewRayContext spx_ray_ctx = ViewRayContext::from_uv_and_depth(spx_uv, spx_packed.depth);

    const float spx_depth = spx_packed.depth;
    const float rpx_depth = half_depth_tex[px];
    const float3 hit_ws = spx_packed.ray_hit_offset_ws + spx_ray_ctx.ray_hit_ws();

    const float3 spx_pos_ws = spx_ray_ctx.ray_hit_ws();

    const float3 trace_origin_ws =
        //view_ray_context.ray_hit_ws();
        view_ray_context.biased_secondary_ray_origin_ws();

    const float3 trace_vec = hit_ws - trace_origin_ws;

    if (rt_is_shadowed(
        acceleration_structure,
        new_ray(
            trace_origin_ws,
            normalize(trace_vec),
            0.0,
            min(5 *length(spx_pos_ws - trace_origin_ws), length(trace_vec) * 0.999)
    ))) {
        r.W = 0;
        reservoir_input_tex[px] = r.as_raw();
    }
}
