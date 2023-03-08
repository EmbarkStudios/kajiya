#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../inc/reservoir.hlsl"
#include "../ircache/bindings.hlsl"
#include "near_field_settings.hlsl"
#include "rtdgi_restir_settings.hlsl"
#include "rtdgi_common.hlsl"

[[vk::binding(0)]] Texture2D<float3> half_view_normal_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> candidate_radiance_tex;
[[vk::binding(3)]] Texture2D<float3> candidate_normal_tex;
[[vk::binding(4)]] Texture2D<float4> candidate_hit_tex;
[[vk::binding(5)]] Texture2D<float4> radiance_history_tex;
[[vk::binding(6)]] Texture2D<float3> ray_orig_history_tex;
[[vk::binding(7)]] Texture2D<float4> ray_history_tex;
[[vk::binding(8)]] Texture2D<uint2> reservoir_history_tex;
[[vk::binding(9)]] Texture2D<float4> reprojection_tex;
[[vk::binding(10)]] Texture2D<float4> hit_normal_history_tex;
[[vk::binding(11)]] Texture2D<float4> candidate_history_tex;
[[vk::binding(12)]] Texture2D<float2> rt_invalidity_tex;
[[vk::binding(13)]] RWTexture2D<float4> radiance_out_tex;
[[vk::binding(14)]] RWTexture2D<float3> ray_orig_output_tex;
[[vk::binding(15)]] RWTexture2D<float4> ray_output_tex;
[[vk::binding(16)]] RWTexture2D<float4> hit_normal_output_tex;
[[vk::binding(17)]] RWTexture2D<uint2> reservoir_out_tex;
[[vk::binding(18)]] RWTexture2D<float4> candidate_out_tex;
[[vk::binding(19)]] RWTexture2D<uint4> temporal_reservoir_packed_tex;
[[vk::binding(20)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

static const float SKY_DIST = 1e4;

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    float3 out_value;
    float3 hit_normal_ws;
    float inv_pdf;
    //bool prev_sample_valid;
};

TraceResult do_the_thing(uint2 px, inout uint rng, RayDesc outgoing_ray, float3 primary_hit_normal) {
    const float4 candidate_radiance_inv_pdf = candidate_radiance_tex[px];
    TraceResult result;
    result.out_value = candidate_radiance_inv_pdf.rgb;
    result.inv_pdf = 1;
    result.hit_normal_ws = direction_view_to_world(candidate_normal_tex[px]);
    return result;
}

int2 get_rpx_offset(uint sample_i, uint frame_index) {
    const int2 offsets[4] = {
        int2(-1, -1),
        int2(1, 1),
        int2(-1, 1),
        int2(1, -1),
    };

    const int2 reservoir_px_offset_base =
        offsets[frame_index & 3]
        + offsets[(sample_i + (frame_index ^ 1)) & 3];

    return
        select(sample_i == 0
        , 0
        , int2(reservoir_px_offset_base))
        ;
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    const int2 hi_px_offset = HALFRES_SUBSAMPLE_OFFSET;
    const uint2 hi_px = px * 2 + hi_px_offset;
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        radiance_out_tex[px] = float4(0.0.xxx, -SKY_DIST);
        hit_normal_output_tex[px] = 0.0.xxxx;
        reservoir_out_tex[px] = 0;
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_biased_depth(uv, depth);
    const float3 normal_vs = half_view_normal_tex[px];
    const float3 normal_ws = direction_view_to_world(normal_vs);
    const float3x3 tangent_to_world = build_orthonormal_basis(normal_ws);
    const float3 refl_ray_origin_ws = view_ray_context.biased_secondary_ray_origin_ws_with_normal(normal_ws);

    const float3 hit_offset_ws = candidate_hit_tex[px].xyz;
    float3 outgoing_dir = normalize(hit_offset_ws);

    uint rng = hash3(uint3(px, frame_constants.frame_index));

    uint2 src_px_sel = px;
    float3 radiance_sel = 0;
    float3 ray_orig_sel_ws = 0;
    float3 ray_hit_sel_ws = 1;
    float3 hit_normal_sel = 1;
    //bool prev_sample_valid = false;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState::create();
    Reservoir1spp reservoir = Reservoir1spp::create();
    const uint reservoir_payload = px.x | (px.y << 16);

    if (is_rtdgi_tracing_frame()) {
        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = refl_ray_origin_ws;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = SKY_DIST;

        const float hit_t = length(hit_offset_ws);

        TraceResult result = do_the_thing(px, rng, outgoing_ray, normal_ws);

        /*if (USE_SPLIT_RT_NEAR_FIELD) {
            const float NEAR_FIELD_FADE_OUT_END = -view_ray_context.ray_hit_vs().z * (SSGI_NEAR_FIELD_RADIUS * gbuffer_tex_size.w * 0.5);
            const float NEAR_FIELD_FADE_OUT_START = NEAR_FIELD_FADE_OUT_END * 0.5;
            float infl = hit_t / (SSGI_NEAR_FIELD_RADIUS * gbuffer_tex_size.w * 0.5) / -view_ray_context.ray_hit_vs().z;
            result.out_value *= smoothstep(0.0, 1.0, infl);
        }*/

        const float p_q = 1.0
            * max(0, sRGB_to_luminance(result.out_value))
            // Note: using max(0, dot) reduces noise in easy areas,
            // but then increases it in corners by undersampling grazing angles.
            // Effectively over time the sampling turns cosine-distributed, which
            // we avoided doing in the first place.
            * step(0, dot(outgoing_dir, normal_ws))
            ;

        const float inv_pdf_q = result.inv_pdf;

        radiance_sel = result.out_value;
        ray_orig_sel_ws = outgoing_ray.Origin;
        ray_hit_sel_ws = outgoing_ray.Origin + outgoing_ray.Direction * hit_t;
        hit_normal_sel = result.hit_normal_ws;
        //prev_sample_valid = result.prev_sample_valid;

        reservoir.init_with_stream(p_q, inv_pdf_q, stream_state, reservoir_payload);

        float rl = lerp(candidate_history_tex[px].y, sqrt(hit_t), 0.05);
        candidate_out_tex[px] = float4(sqrt(hit_t), rl, 0, 0);
    }

    const float rt_invalidity = sqrt(saturate(rt_invalidity_tex[px].y));

    const bool use_resampling = DIFFUSE_GI_USE_RESTIR;
    //const bool use_resampling = false;

    // 1 (center) plus offset samples
    const uint MAX_RESOLVE_SAMPLE_COUNT =
        select(RESTIR_TEMPORAL_USE_PERMUTATIONS
        , 5
        , 1);

    float center_M = 0;

    if (use_resampling) {
        for (
            uint sample_i = 0;
            sample_i < MAX_RESOLVE_SAMPLE_COUNT
            // Use permutation sampling, but only up to a certain M; those are lower quality,
            // so we want to be rather conservative.
            && stream_state.M_sum < 1.25 * RESTIR_TEMPORAL_M_CLAMP;
            ++sample_i) {
            const int2 rpx_offset = get_rpx_offset(sample_i, frame_constants.frame_index);
            if (sample_i > 0 && all(rpx_offset == 0)) {
                // No point using the center sample twice
                continue;
            }

            const float4 reproj = reprojection_tex[hi_px + rpx_offset * 2];

            // Can't use linear interpolation, but we can interpolate stochastically instead
            //const float2 reproj_rand_offset = float2(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))) - 0.5;
            // Or not at all.
            const float2 reproj_rand_offset = 0.0;

            const uint2 xor_seq[4] = {
                uint2(3, 3),
                uint2(2, 1),
                uint2(1, 2),
                uint2(3, 3),
            };
            const uint2 permutation_xor_val =
                xor_seq[frame_constants.frame_index & 3];            

            const int2 permuted_reproj_px = floor(
                select(sample_i == 0
                    , px
                    // My poor approximation of permutation sampling.
                    // https://twitter.com/more_fps/status/1457749362025459715
                    //
                    // When applied everywhere, it does nicely reduce noise, but also makes the GI less reactive
                    // since we're effectively increasing the lifetime of the most attractive samples.
                    // Where it does come in handy though is for boosting convergence rate for newly revealed
                    // locations.
                    , ((px + rpx_offset) ^ permutation_xor_val))
                + gbuffer_tex_size.xy * reproj.xy * 0.5 + reproj_rand_offset + 0.5);

            const int2 rpx = permuted_reproj_px + rpx_offset;
            const uint2 rpx_hi = rpx * 2 + hi_px_offset;

            const int2 permuted_neighbor_px = floor(
                select(sample_i == 0
                    , px
                    // ditto
                    , ((px + rpx_offset) ^ permutation_xor_val)) + 0.5);

            const int2 neighbor_px = permuted_neighbor_px + rpx_offset;
            const uint2 neighbor_px_hi = neighbor_px * 2 + hi_px_offset;

            // WRONG. needs previous normal
            // const float3 sample_normal_vs = half_view_normal_tex[rpx];
            // // Note: also doing this for sample 0, as under extreme aliasing,
            // // we can easily get bad samples in.
            // if (dot(sample_normal_vs, normal_vs) < 0.7) {
            //     continue;
            // }

            Reservoir1spp r = Reservoir1spp::from_raw(reservoir_history_tex[rpx]);
            const uint2 spx = reservoir_payload_to_px(r.payload);

            float visibility = 1;
            //float relevance = select(sample_i == 0, 1, 0.5);
            float relevance = 1;

            //const float2 sample_uv = get_uv(rpx_hi, gbuffer_tex_size);
            const float sample_depth = depth_tex[neighbor_px_hi];

            // WRONG: needs previous depth
            // if (length(prev_ray_orig_and_dist.xyz - refl_ray_origin_ws) > 0.1 * -view_ray_context.ray_hit_vs().z) {
            //     // Reject disocclusions
            //     continue;
            // }

            const float3 prev_ray_orig = ray_orig_history_tex[spx];
            if (length(prev_ray_orig - refl_ray_origin_ws) > 0.1 * -view_ray_context.ray_hit_vs().z) {
                // Reject disocclusions
                continue;
            }

            // Note: also doing this for sample 0, as under extreme aliasing,
            // we can easily get bad samples in.
            if (0 == sample_depth) {
                continue;
            }

            // TODO: some more rejection based on the reprojection map.
            // This one is not enough ("battle", buttom of tower).
            if (reproj.z == 0) {
                continue;
            }

            #if 1
                relevance *= 1 - smoothstep(0.0, 0.1, inverse_depth_relative_diff(depth, sample_depth));
            #else
                if (inverse_depth_relative_diff(depth, sample_depth) > 0.2) {
                    continue;
                }
            #endif

            const float3 sample_normal_vs = half_view_normal_tex[neighbor_px].rgb;
            const float normal_similarity_dot = max(0.0, dot(sample_normal_vs, normal_vs));

        // Increases noise, but prevents leaking in areas of geometric complexity
        #if 1
            // High cutoff seems unnecessary. Had it at 0.9 before.
            const float normal_cutoff = 0.2;
            if (sample_i != 0 && normal_similarity_dot < normal_cutoff) {
                continue;
            }
        #endif

            relevance *= pow(normal_similarity_dot, 4);

            // TODO: this needs fixing with reprojection
            //const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);

            const float4 sample_hit_ws_and_dist = ray_history_tex[spx] + float4(prev_ray_orig, 0.0);
            const float3 sample_hit_ws = sample_hit_ws_and_dist.xyz;
            //const float3 prev_dir_to_sample_hit_unnorm_ws = sample_hit_ws - sample_ray_ctx.ray_hit_ws();
            //const float3 prev_dir_to_sample_hit_ws = normalize(prev_dir_to_sample_hit_unnorm_ws);
            const float prev_dist = sample_hit_ws_and_dist.w;

            // Note: `hit_normal_history_tex` is not reprojected.
            const float4 sample_hit_normal_ws_dot = decode_hit_normal_and_dot(hit_normal_history_tex[spx]);

            /*if (sample_i > 0 && !(prev_dist > 1e-4)) {
                continue;
            }*/

            const float3 dir_to_sample_hit_unnorm = sample_hit_ws - refl_ray_origin_ws;
            const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
            const float3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

            const float center_to_hit_vis = -dot(sample_hit_normal_ws_dot.xyz, dir_to_sample_hit);
            //const float prev_to_hit_vis = -dot(sample_hit_normal_ws_dot.xyz, prev_dir_to_sample_hit_ws);

            const float4 prev_rad =
                radiance_history_tex[spx]
                * float4((frame_constants.pre_exposure_delta).xxx, 1);

            // From the ReSTIR paper:
            // With temporal reuse, the number of candidates M contributing to the
            // pixel can in theory grow unbounded, as each frame always combines
            // its reservoir with the previous frame’s. This causes (potentially stale)
            // temporal samples to be weighted disproportionately high during
            // resampling. To fix this, we simply clamp the previous frame’s M
            // to at most 20× of the current frame’s reservoir’s M

            r.M = max(0, min(r.M, exp2(log2(RESTIR_TEMPORAL_M_CLAMP) * (1.0 - rt_invalidity))));
            //r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP);
            //r.M = min(r.M, 0.1);

            const float p_q = 1
                * max(0, sRGB_to_luminance(prev_rad.rgb))
                // Note: using max(0, dot) reduces noise in easy areas,
                // but then increases it in corners by undersampling grazing angles.
                // Effectively over time the sampling turns cosine-distributed, which
                // we avoided doing in the first place.
                * step(0, dot(dir_to_sample_hit, normal_ws))
                ;

            float jacobian = 1;

            // Note: needed for sample 0 too due to temporal jitter.
            {
                // Distance falloff. Needed to avoid leaks.
                jacobian *= clamp(prev_dist / dist_to_sample_hit, 1e-4, 1e4);
                jacobian *= jacobian;

                // N of hit dot -L. Needed to avoid leaks. Without it, light "hugs" corners.
                //
                jacobian *= clamp(center_to_hit_vis / sample_hit_normal_ws_dot.w, 0, 1e4);
            }

            // Fixes boiling artifacts near edges. Unstable jacobians,
            // but also effectively reduces reliance on reservoir exchange
            // in tight corners, which is desirable since the well-distributed
            // raw samples thrown at temporal filters will do better.
            if (RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION) {
                // Clamp neighbors give us a hit point that's considerably easier to sample
                // from our own position than from the neighbor. This can cause some darkening,
                // but prevents fireflies.
                //
                // The darkening occurs in corners, where micro-bounce should be happening instead.

                #if 1
                    // Doesn't over-darken corners as much
                    jacobian = min(jacobian, RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE);
                #else
                    // Slightly less noise
                    if (jacobian > RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE) { continue; }
                #endif
            }

            r.M *= relevance;

            if (0 == sample_i) {
                center_M = r.M;
            }

            if (reservoir.update_with_stream(
                r, p_q, jacobian * visibility,
                stream_state, reservoir_payload, rng
            )) {
                outgoing_dir = dir_to_sample_hit;
                src_px_sel = rpx;
                radiance_sel = prev_rad.rgb;
                ray_orig_sel_ws = prev_ray_orig;
                ray_hit_sel_ws = sample_hit_ws;
                hit_normal_sel = sample_hit_normal_ws_dot.xyz;
            }
        }

        reservoir.finish_stream(stream_state);
        reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);
    }

    // TODO: this results in M being accumulated at a slower rate, although finally reaching
    // the limit we're after. What it does is practice is slow down the kernel tightening
    // in the subsequent spatial reservoir resampling.
    reservoir.M = center_M + 0.5;
    //reservoir.M = center_M + 1;

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = refl_ray_origin_ws;
    outgoing_ray.TMin = 0;

    const float4 hit_normal_ws_dot = float4(hit_normal_sel, -dot(hit_normal_sel, outgoing_ray.Direction));

    radiance_out_tex[px] = float4(radiance_sel, dot(normal_ws, outgoing_ray.Direction));
    ray_orig_output_tex[px] = ray_orig_sel_ws;
    hit_normal_output_tex[px] = encode_hit_normal_and_dot(hit_normal_ws_dot);
    ray_output_tex[px] = float4(ray_hit_sel_ws - ray_orig_sel_ws, length(ray_hit_sel_ws - refl_ray_origin_ws));
    reservoir_out_tex[px] = reservoir.as_raw();

    TemporalReservoirOutput res_packed;
    res_packed.depth = depth;
    res_packed.ray_hit_offset_ws = ray_hit_sel_ws - view_ray_context.ray_hit_ws();
    res_packed.luminance = max(0.0, sRGB_to_luminance(radiance_sel));
    res_packed.hit_normal_ws = hit_normal_ws_dot.xyz;
    temporal_reservoir_packed_tex[px] = res_packed.as_raw();
}
