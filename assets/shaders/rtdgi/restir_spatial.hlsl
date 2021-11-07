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
[[vk::binding(1)]] Texture2D<float4> hit_normal_tex;
[[vk::binding(2)]] Texture2D<float4> ray_tex;
[[vk::binding(3)]] Texture2D<float4> reservoir_input_tex;
[[vk::binding(4)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(5)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(6)]] Texture2D<float> half_depth_tex;
[[vk::binding(7)]] Texture2D<float4> ssao_tex;
[[vk::binding(8)]] Texture2D<float4> candidate_input_tex;
[[vk::binding(9)]] RWTexture2D<float4> reservoir_output_tex;
[[vk::binding(10)]] cbuffer _ {
    float4 gbuffer_tex_size;
    float4 output_tex_size;
    uint spatial_reuse_pass_idx;
};

static const float CENTER_SAMPLE_M_TRUNCATION = 0.2;

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

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float3 center_normal_vs = half_view_normal_tex[px].rgb;
    const float3 center_normal_ws = direction_view_to_world(center_normal_vs);
    const float center_depth = half_depth_tex[px];
    const float center_ssao = ssao_tex[px * 2].r;

    Reservoir1spp reservoir = Reservoir1spp::create();
    float p_q_sel = 0;
    float3 dir_sel = 1;
    float M_sum = 0;

    static const float GOLDEN_ANGLE = 2.39996323;

    // TODO: split off into a separate temporal stage, following ReSTIR GI
    uint sample_count = DIFFUSE_GI_USE_RESTIR ? 8 : 1;
    float sample_radius_offset = uint_to_u01_float(hash1_mut(rng));

    float poor_normals = 0;

    Reservoir1spp center_r = Reservoir1spp::from_raw(reservoir_input_tex[px]);

    //float radius_mult = spatial_reuse_pass_idx == 0 ? 1.5 : 3.5;

    float kernel_radius = lerp(2.0, 16.0, ssao_tex[hi_px].r);
    if (spatial_reuse_pass_idx == 1) {
        sample_count = 5;
        kernel_radius = lerp(2.0, 32.0, ssao_tex[hi_px].r);
    }

    const uint TARGET_M = 512;

    // Scrambling angles here would be nice, but results in bad cache thrashing.
    // Quantizing the offsets results in mild cache abuse, and fixes most of the artifacts
    // (flickering near edges, e.g. under sofa in the UE5 archviz apartment scene).
    const float ang_offset = uint_to_u01_float(hash3(
        uint3((px >> 2), frame_constants.frame_index * 2 + spatial_reuse_pass_idx)
    )) * M_PI * 2;

    //sample_count = 1;

    uint valid_sample_count = 0;
    for (uint sample_i = 0; sample_i < sample_count && M_sum < TARGET_M; ++sample_i) {
        float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
        float radius = 0 == sample_i ? 0 : float(sample_i + sample_radius_offset) * (kernel_radius / sample_count);
        int2 rpx_offset = float2(cos(ang), sin(ang)) * radius;

        const bool is_center_sample = sample_i == 0;
        //const bool is_center_sample = all(rpx_offset == 0);

        const int2 rpx = px + rpx_offset;
        Reservoir1spp r = Reservoir1spp::from_raw(reservoir_input_tex[rpx]);

        if (is_center_sample && spatial_reuse_pass_idx == 0) {
            #if 1
                // I don't quite understand how, but suppressing the central reservoir's
                // sample count here reduces noise, especially reducing fireflies,
                // and yet has a pretty small impact on the sharpness of the output.
                // A decent value seems to be around 20% of the limit
                // in the preceding exchange pass.
                r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP * CENTER_SAMPLE_M_TRUNCATION);
            #else
                r.M = min(r.M, 500);
            #endif
        } else {
            // After the ReSTIR GI paper
            r.M = min(r.M, 500);
        }

        const uint2 spx = reservoir_payload_to_px(r.payload);
        float4 prev_irrad = irradiance_tex[spx];
        float visibility = 1;

        const int2 sample_offset = int2(px) - int2(rpx);
        const float sample_dist2 = dot(sample_offset, sample_offset);
        const float3 sample_normal_vs = half_view_normal_tex[rpx].rgb;

        #if DIFFUSE_GI_BRDF_SAMPLING
            float normal_cutoff = 0.9;
        #else
            float normal_cutoff = 0.5;
        #endif
        
        if (spatial_reuse_pass_idx != 0) {
            normal_cutoff = 0.5;
        }

        if (center_r.M < 10) {
            normal_cutoff = 0.5 * exp2(-max(0, poor_normals) * 0.3);
        }

        // Note: Waaaaaay more loose than the ReSTIR papers. Reduces noise in
        // areas of high geometric complexity. The resulting bias tends to brighten edges,
        // and we clamp that effect later. The artifacts is less prounounced normal map detail.
        // TODO: detect this first, and sharpen the threshold. The poor normal counting below
        // is a shitty take at that.
        const float normal_similarity_dot = dot(sample_normal_vs, center_normal_vs);
        if (!is_center_sample && normal_similarity_dot < normal_cutoff) {
            poor_normals += 1;
            continue;
        } else {
            poor_normals -= 1;
        }

        const float ssao_dart = uint_to_u01_float(hash1_mut(rng));
        const float sample_ssao = ssao_tex[spx * 2 + hi_px_subpixels[frame_constants.frame_index & 3]].r;

        // Balance between details and splotches in corner
        const float ssao_threshold = spatial_reuse_pass_idx == 0 ? 0.2 : 0.4;

        // Was: abs(sample_ssao - center_ssao); that can however reject too aggressively.
        // Some leaking is better than flicker and very dark corners.
        const float ssao_infl = smoothstep(0.0, ssao_threshold, abs(sample_ssao - center_ssao));

        if (!is_center_sample && ssao_infl > ssao_dart) {
            // Note: improves contacts, but results in boiling/noise in corners
            // This is really just an approximation of a visbility check,
            // which we can do in a better way.
            continue;
        }

        const float2 sample_uv = get_uv(
            rpx * 2 + hi_px_subpixels[frame_constants.frame_index & 3],
            gbuffer_tex_size);
        const float sample_depth = half_depth_tex[rpx];
        
        if (sample_depth == 0.0) {
            continue;
        }

        const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);
        const float3 sample_origin_vs = sample_ray_ctx.ray_hit_vs();

        const float4 sample_hit_ws_and_dist = ray_tex[spx] + float4(get_eye_position(), 0.0);
        const float3 sample_hit_ws = sample_hit_ws_and_dist.xyz;
        const float3 prev_dir_to_sample_hit_unnorm_ws = sample_hit_ws - sample_ray_ctx.ray_hit_ws();
        const float3 prev_dir_to_sample_hit_ws = normalize(prev_dir_to_sample_hit_unnorm_ws);
        const float prev_dist = length(prev_dir_to_sample_hit_unnorm_ws);

        // Reject hits too close to the surface
        if (!is_center_sample && !(prev_dist > 1e-8)) {
            continue;
        }

        const float3 dir_to_sample_hit_unnorm = sample_hit_ws - view_ray_context.biased_secondary_ray_origin_ws();
        const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
        const float3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

        // Reject hits below the normal plane
        if (!is_center_sample && dot(dir_to_sample_hit, center_normal_ws) < 1e-5) {
            continue;
        }

        // TODO: combine all those into a single similarity metric?

        // Reject neighbors with vastly different depths
        if (spatial_reuse_pass_idx == 0) {
            if (!is_center_sample && abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)) > 0.1) {
                continue;
            }
        } else {
            if (!is_center_sample && abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)) > 0.2) {
                continue;
            }
        }

        // Approx shadowing
        {
    		const float3 surface_offset_vs = sample_origin_vs - view_ray_context.ray_hit_vs();
            const float sample_inclination = dot(normalize(surface_offset_vs), center_normal_vs);
            const float ray_inclination = dot(dir_to_sample_hit, center_normal_ws);

            if (!is_center_sample && ray_inclination * 0.2 < sample_inclination) {
                continue;
            }

            // Raymarch to check occlusion
            if (!is_center_sample) {
                #if 0
                    const int k_count = 3;
                    const int range_px = k_count * 3;   // Note: causes flicker if smaller

                    for (int k = 0; k < k_count; ++k) {
                        float3 dir_to_sample_hit_vs = direction_world_to_view(dir_to_sample_hit);
                        float2 intermediary_sample_uv = uv + normalize(dir_to_sample_hit_vs.xy) * float2(1, -1) * output_tex_size.zw * (k + 0.5) / k_count * range_px;
                        const int2 intermediary_sample_px = int2(floor(intermediary_sample_uv * output_tex_size.xy));
                        intermediary_sample_uv = (intermediary_sample_px + 0.5) * output_tex_size.zw;

                        const float intermediary_sample_depth = half_depth_tex[intermediary_sample_px];
                        if (intermediary_sample_depth == 0) {
                            continue;
                        }

                        const ViewRayContext intermediary_sample_ray_ctx = ViewRayContext::from_uv_and_depth(intermediary_sample_uv, intermediary_sample_depth);
                        const float3 intermediary_surface_offset_vs = intermediary_sample_ray_ctx.ray_hit_vs() - view_ray_context.ray_hit_vs();
                        if (length(intermediary_surface_offset_vs) > length(surface_offset_vs)) {
                            continue;
                        }

                        if (dot(dir_to_sample_hit_vs, intermediary_surface_offset_vs) > dist_to_sample_hit) {
                            //continue;
                        }

                        const float intermediary_sample_inclination = dot(normalize(intermediary_surface_offset_vs), center_normal_vs);
                        visibility *= ray_inclination > intermediary_sample_inclination;
                    }
                #else
                    // TODO: finish the derivations, don't perspective-project for every sample.

                    const float3 raymarch_dir_unnorm_ws = sample_hit_ws - view_ray_context.ray_hit_ws();
                    const float3 raymarch_end_ws =
                        view_ray_context.ray_hit_ws()
                        // TODO: what's a good max distance to raymarch? Probably need to project some stuff
                        + raymarch_dir_unnorm_ws * min(1.0, length(surface_offset_vs) / length(raymarch_dir_unnorm_ws));

                    const float2 raymarch_end_uv = cs_to_uv(position_world_to_clip(raymarch_end_ws).xy);
                    const float2 raymarch_len_px = (raymarch_end_uv - uv) * output_tex_size.xy;

                    const int k_count = min(3, int(floor(length(raymarch_len_px) / 3)));

                    // Depth values only have the front; assume a certain thickness.
                    const float Z_LAYER_THICKNESS = 0.03;

                    for (int k = 0; k < k_count; ++k) {
                        const float t = (k + 0.5) / k_count;
                        const float3 interp_pos_ws = lerp(view_ray_context.ray_hit_ws(), raymarch_end_ws, t);
                        const float3 interp_pos_cs = position_world_to_clip(interp_pos_ws);
                        const float depth_at_interp = half_depth_tex.SampleLevel(sampler_nnc, cs_to_uv(interp_pos_cs.xy), 0);
                        if (depth_at_interp > interp_pos_cs.z
                            && inverse_depth_relative_diff(interp_pos_cs.z, depth_at_interp) < Z_LAYER_THICKNESS
                        ) {
                            visibility = 0;
                        }
                    }
                #endif
    		}
        }

        const float4 sample_hit_normal_ws_dot = hit_normal_tex[spx];

        float p_q = 1;
        p_q *= max(1e-3, calculate_luma(prev_irrad.rgb));

        // Actually looks more noisy with this the N dot L when using BRDF sampling.
        // With (hemi)spherical sampling, it's fine.
        #if !DIFFUSE_GI_BRDF_SAMPLING
            p_q *= max(0, dot(dir_to_sample_hit, center_normal_ws));
        #endif

        float jacobian = 1;

        // Distance falloff. Needed to avoid leaks.
        jacobian *= max(0.0, prev_dist) / max(1e-4, dist_to_sample_hit);
        jacobian *= jacobian;

        // N of hit dot -L. Needed to avoid leaks.
        jacobian *=
            max(0.0, -dot(sample_hit_normal_ws_dot.xyz, dir_to_sample_hit))
            /// max(1e-5, sample_hit_normal_ws_dot.w);
            / max(1e-5, -dot(sample_hit_normal_ws_dot.xyz, prev_dir_to_sample_hit_ws));

        #if DIFFUSE_GI_BRDF_SAMPLING
            // N dot L. Useful for normal maps, micro detail.
            // The min(const, _) should not be here, but it prevents fireflies and brightening of edges
            // when we don't use a harsh normal cutoff to exchange reservoirs with.
            //jacobian *= min(1.2, max(0.0, prev_irrad.a) / dot(dir_to_sample_hit, center_normal_ws));
            //jacobian *= max(0.0, prev_irrad.a) / dot(dir_to_sample_hit, center_normal_ws);
        #endif

        if (is_center_sample) {
            jacobian = 1;
        }

        if (!(p_q > 0)) {
            continue;
        }

        float w = p_q * r.W * r.M;
        if (reservoir.update(w * jacobian * visibility, r.payload, rng)) {
            p_q_sel = p_q;
            dir_sel = dir_to_sample_hit;
        }

        M_sum += r.M;
        valid_sample_count += 1;
    }

    reservoir.M = M_sum;
    reservoir.W =
        (1.0 / max(1e-8, p_q_sel))
        * (reservoir.w_sum / max(CENTER_SAMPLE_M_TRUNCATION, reservoir.M));

    // (Source of bias?) suppress fireflies
    // Unclear what kind of bias. When clamped lower, e.g. 1.0,
    // the whole scene gets darker.
    reservoir.W = min(reservoir.W, 5);

    reservoir_output_tex[px] = reservoir.as_raw();
}
