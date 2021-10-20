#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/reservoir.hlsl"
#include "restir_settings.hlsl"

[[vk::binding(0)]] Texture2D<float4> irradiance_tex;
[[vk::binding(1)]] Texture2D<float4> ray_tex;
[[vk::binding(2)]] Texture2D<float4> reservoir_input_tex;
[[vk::binding(3)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(4)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(5)]] Texture2D<float> half_depth_tex;
[[vk::binding(6)]] Texture2D<float4> ssao_tex;
[[vk::binding(7)]] RWTexture2D<float4> reservoir_output_tex;
[[vk::binding(8)]] RWTexture2D<float4> irradiance_output_tex;
[[vk::binding(9)]] cbuffer _ {
    float4 gbuffer_tex_size;
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
    uint spatial_reuse_pass_idx;
};

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    const uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };
    const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];
    float depth = half_depth_tex[px];

    const uint seed = frame_constants.frame_index + spatial_reuse_pass_idx * 123;
    uint rng = hash3(uint3(px, seed));

    if (0.0 == depth) {
        irradiance_output_tex[px] = 0.0.xxxx;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    const float3 center_normal_vs = half_view_normal_tex[px].rgb;
    const float center_depth = half_depth_tex[px];

    //Reservoir1spp reservoir = Reservoir1spp::from_raw(reservoir_input_tex[px]);
    Reservoir1spp reservoir = Reservoir1spp::create();

    float p_q_sel = 0;//reservoir.W;//calculate_luma(irradiance);
    // p_q_sel *= max(0, dot(prev_dir, gbuffer.normal));

    float jacobian_correction = 1;

    float M_sum = reservoir.M;

    static const float GOLDEN_ANGLE = 2.39996323;

    // TODO: split off into a separate temporal stage, following ReSTIR GI
    const uint sample_count = DIFFUSE_GI_USE_RESTIR ? 8 : 1;
    float sample_radius_offset = uint_to_u01_float(hash1_mut(rng));

    const float ang_offset = uint_to_u01_float(hash1_mut(rng)) * M_PI * 2;
    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
        float radius = 0 == sample_i ? 0 : float(sample_i + sample_radius_offset) * 2.5;
        int2 sample_offset = float2(cos(ang), sin(ang)) * radius;

        const int2 reservoir_px = px + sample_offset;
        Reservoir1spp r = Reservoir1spp::from_raw(reservoir_input_tex[reservoir_px]);

        // After the ReSTIR GI paper
        r.M = min(r.M, 500);

        const uint2 spx = reservoir_payload_to_px(r.payload);

        float4 sample_gbuffer_packed = gbuffer_tex[spx * 2];
        GbufferData sample_gbuffer = GbufferDataPacked::from_uint4(asuint(sample_gbuffer_packed)).unpack();

        if (dot(sample_gbuffer.normal, gbuffer.normal) < 0.9) {
            continue;
        }

        const float4 prev_hit_ws_and_dist = ray_tex[spx];
        const float3 prev_hit_ws = prev_hit_ws_and_dist.xyz;
        const float prev_dist = prev_hit_ws_and_dist.w;

        // Reject hits too close to the surface
        if (!(prev_dist > 1e-8)) {
            continue;
        }

        const float3 prev_dir_unnorm = prev_hit_ws - view_ray_context.biased_secondary_ray_origin_ws();
        const float prev_dist_now = length(prev_dir_unnorm);
        const float3 prev_dir = normalize(prev_dir_unnorm);

        // Reject hits below the normal plane
        if (dot(prev_dir, gbuffer.normal) < 1e-3) {
            continue;
        }

        const float3 sample_normal_vs = half_view_normal_tex[spx].rgb;
        const float sample_depth = half_depth_tex[spx];

        // Reject neighbors with vastly different depths
        if (abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)) > 0.1) {
            continue;
        }        

        const float4 prev_irrad = irradiance_tex[spx];

        float p_q = 1;
        p_q *= calculate_luma(prev_irrad.rgb);
        p_q *= max(0, dot(prev_dir, gbuffer.normal));

        //float sample_jacobian_correction = 1.0 / max(1e-4, prev_dist);
        //float sample_jacobian_correction = 1;
        float sample_jacobian_correction = max(0.0, prev_dist) / max(1e-4, prev_dist_now);
        sample_jacobian_correction *= sample_jacobian_correction;

        sample_jacobian_correction *= max(0.0, prev_irrad.a) / dot(prev_dir, gbuffer.normal);
        //sample_jacobian_correction = 1;

        //p_q *= sample_jacobian_correction;

        if (!(p_q > 0)) {
            continue;
        }

        float w = p_q * r.W * r.M;
        if (reservoir.update(w, r.payload, rng)) {
            p_q_sel = p_q;

            // TODO; seems wrong.
            jacobian_correction = sample_jacobian_correction;
        }

        M_sum += r.M;
    }

    reservoir.M = M_sum;
    reservoir.W = max(1e-5, reservoir.w_sum / (p_q_sel * reservoir.M));

    reservoir_output_tex[px] = reservoir.as_raw();

    #if 1
        irradiance_output_tex[px] = float4(
            irradiance_tex[reservoir_payload_to_px(reservoir.payload)].rgb * reservoir.W
            // HAAAAAACK; the min prevents fireflies :shrug:
            * min(1, jacobian_correction),
            //reservoir.W.xxx * 0.1,
            1
        );
        return;
    #endif
}
