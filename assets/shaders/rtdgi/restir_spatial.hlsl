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

    //Reservoir1spp reservoir = Reservoir1spp::from_raw(reservoir_input_tex[px]);
    Reservoir1spp reservoir = Reservoir1spp::create();

    float p_q_sel = 0;//reservoir.W;//calculate_luma(irradiance);
    float3 dir_sel = 1;
    // p_q_sel *= max(0, dot(prev_dir, center_normal_ws));

    float M_sum = reservoir.M;

    static const float GOLDEN_ANGLE = 2.39996323;

    // TODO: split off into a separate temporal stage, following ReSTIR GI
    uint sample_count = DIFFUSE_GI_USE_RESTIR ? 8 : 1;
    float sample_radius_offset = uint_to_u01_float(hash1_mut(rng));

    float poor_normals = 0;

    Reservoir1spp center_r = Reservoir1spp::from_raw(reservoir_input_tex[px]);

    //float radius_mult = spatial_reuse_pass_idx == 0 ? 1.5 : 3.5;

    // Note: pow(ssao, n) is a balance between details and splotches in corner
    float kernel_radius = lerp(2.0, 12.0, pow(ssao_tex[hi_px].r, 2));
    if (spatial_reuse_pass_idx == 0 && center_r.M < 10) {
        kernel_radius = 16.0;
    }

    uint valid_sample_count = 0;
    uint valid_sample_count_limit = 5;

    if (spatial_reuse_pass_idx == 1) {
        //sample_count = 5;
        sample_count = 5;
        valid_sample_count_limit = 3;
        kernel_radius = lerp(8.0, 24.0, pow(ssao_tex[hi_px].r, 4));
    }

    /*const int2 pass1_offsets[5] = {
        int2(0, 0),
        int2(16, 0),
        int2(-16, 0),
        int2(0, 16),
        int2(0, -16),
    };*/

    const float ang_offset = uint_to_u01_float(hash1_mut(rng)) * M_PI * 2;
    // early out saves a lot of comp, but leaves noise in tricky cases (old building interior)
    //for (uint sample_i = 0; sample_i < 16 && valid_sample_count < 8; ++sample_i) {
    for (uint sample_i = 0; sample_i < sample_count && valid_sample_count < valid_sample_count_limit; ++sample_i) {
        float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
        float radius = 0 == sample_i ? 0 : float(sample_i + sample_radius_offset) * (kernel_radius / valid_sample_count_limit);
        //float radius = float(sample_i + sample_radius_offset) * (kernel_radius / valid_sample_count_limit);
        int2 rpx_offset = float2(cos(ang), sin(ang)) * radius;
        //const bool is_center_sample = sample_i == 0;
        const bool is_center_sample = all(rpx_offset == 0);

        /*if (spatial_reuse_pass_idx == 1) {
            rpx_offset = pass1_offsets[sample_i];
        }*/

        const int2 rpx = px + rpx_offset;
        Reservoir1spp r = Reservoir1spp::from_raw(reservoir_input_tex[rpx]);

        if (is_center_sample) {
            #if 1
                // I don't quite understand how, but strongly suppressing
                // the central reservoir's sample count here reduces noise,
                // especially reducing fireflies, and yet has little effect
                // on the blurriness of the output.
                // A decent value seems to be around 10% of the limit
                // in the preceding exchange pass.
                r.M = min(r.M, RESTIR_TEMPORAL_M_CLAMP * 0.1);
            #else
                r.M = min(r.M, 500);
            #endif
        } else {
            // After the ReSTIR GI paper
            r.M = min(r.M, 500);
        }

        const uint2 spx = reservoir_payload_to_px(r.payload);

        const int2 sample_offset = int2(px) - int2(rpx);
        const float sample_dist2 = dot(sample_offset, sample_offset);
        const float3 sample_normal_vs = half_view_normal_tex[rpx].rgb;

        float normal_cutoff = 0.9;
        if (spatial_reuse_pass_idx == 1) {
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
        const float ssao_infl = smoothstep(0.0, 0.2, max(1, poor_normals) * abs(sample_ssao - center_ssao));
        // TODO: make threshold adaptive
        //if (!is_center_sample && abs(sample_ssao - center_ssao) > 0.1 * max(1, poor_normals)) {

        if (spatial_reuse_pass_idx == 0) {
            if (!is_center_sample && ssao_infl > ssao_dart) {
                // Note: improves contacts, but results in boiling/noise in corners
                // This is really just an approximation of a visbility check,
                // which we can do in a better way.
                //continue;
            }
        }

        const float2 sample_uv = get_uv(
            rpx * 2 + hi_px_subpixels[frame_constants.frame_index & 3],
            gbuffer_tex_size);
        const float sample_depth = half_depth_tex[rpx];

        const ViewRayContext sample_ray_ctx = ViewRayContext::from_uv_and_depth(sample_uv, sample_depth);
        const float3 sample_origin_vs = sample_ray_ctx.ray_hit_vs();

        const float4 prev_hit_ws_and_dist = ray_tex[spx];
        const float3 prev_hit_ws = prev_hit_ws_and_dist.xyz;
        const float prev_dist = length(prev_hit_ws - sample_ray_ctx.ray_hit_ws()); //prev_hit_ws_and_dist.w;

#if 1
        if (!is_center_sample && spatial_reuse_pass_idx == 1) {
            /*const float4 own_hit_ws_and_dist = ray_tex[px];
            const float3 own_hit_ws = own_hit_ws_and_dist.xyz;
            const float own_dist = length(own_hit_ws - view_ray_context.ray_hit_ws());

            if (dot(
                normalize(prev_hit_ws - sample_ray_ctx.ray_hit_ws()),
                normalize(own_hit_ws - view_ray_context.ray_hit_ws())
            ) < 0.9) {
                continue;
            } else {
                if (2 * abs(prev_dist - own_dist) / (prev_dist + own_dist) > 10) {
                    continue;
                }
            }*/

            const float a = candidate_input_tex[px].y;
            const float b = candidate_input_tex[rpx].y;
            if (2 * abs(a - b) / (a + b) > 1) {
                continue;
            }
        }
#endif

        // Reject hits too close to the surface
        if (!is_center_sample && !(prev_dist > 1e-8)) {
            continue;
        }

        const float3 prev_dir_unnorm = prev_hit_ws - view_ray_context.biased_secondary_ray_origin_ws();
        const float prev_dist_now = length(prev_dir_unnorm);
        const float3 prev_dir = normalize(prev_dir_unnorm);

        // Reject hits below the normal plane
        if (!is_center_sample && dot(prev_dir, center_normal_ws) < 1e-5) {
            continue;
        }

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
        //if (spatial_reuse_pass_idx == 0)
        {
    		const float3 surface_offset = sample_origin_vs - view_ray_context.ray_hit_vs();
            const float fraction_of_normal_direction_as_offset = dot(surface_offset, center_normal_vs) / length(surface_offset);
            const float wi_z = dot(prev_dir, center_normal_ws);

            if (!is_center_sample && wi_z * 0.2 < fraction_of_normal_direction_as_offset) {
    			continue;
    		}
        }

        // TODO: combine all those into a single similarity metric

        const float4 prev_irrad = irradiance_tex[spx];
        const float4 prev_hit_normal_ws_dot = hit_normal_tex[spx];

        float p_q = 1;
        p_q *= max(1e-3, calculate_luma(prev_irrad.rgb));
        //p_q *= exp2(-sqrt(sample_dist2) * 0.5);

        // Actually looks more noisy with this the N dot L when using BRDF sampling.
        // With (hemi)spherical sampling, it's fine.
        #if !DIFFUSE_GI_BRDF_SAMPLING
            p_q *= max(0, dot(prev_dir, center_normal_ws));
        #endif

        float jacobian = 1;

        // Distance falloff. Needed to avoid leaks.
        jacobian *= max(0.0, prev_dist) / max(1e-4, prev_dist_now);
        jacobian *= jacobian;

        // N of hit dot -L. Needed to avoid leaks.
        jacobian *= max(0.0, -dot(prev_hit_normal_ws_dot.xyz, prev_dir)) / max(1e-4, prev_hit_normal_ws_dot.w);

        // N dot L. Useful for normal maps, micro detail.
        // The min(const, _) should not be here, but it prevents fireflies and brightening of edges
        // when we don't use a harsh normal cutoff to exchange reservoirs with.
        //jacobian *= min(1.2, max(0.0, prev_irrad.a) / dot(prev_dir, center_normal_ws));
        //jacobian *= max(0.0, prev_irrad.a) / dot(prev_dir, center_normal_ws);

        if (is_center_sample) {
            jacobian = 1;
        }

        if (!(p_q > 0)) {
            continue;
        }

        float w = p_q * r.W * r.M;
        if (reservoir.update(w * jacobian, r.payload, rng)) {
            p_q_sel = p_q;
            dir_sel = prev_dir;
        }

        M_sum += r.M;
        valid_sample_count += 1;
    }

    reservoir.M = M_sum;
    reservoir.W = (1.0 / max(1e-5, p_q_sel)) * (reservoir.w_sum / reservoir.M);

    // TODO: find the NaN
    if (!(reservoir.W >= 0)) {
        reservoir = Reservoir1spp::create();
    }

    reservoir_output_tex[px] = reservoir.as_raw();

    /*float3 irradiance = irradiance_tex[reservoir_payload_to_px(reservoir.payload)].rgb;
    float likelihood = calculate_luma(irradiance) * center_r.W;

    #if 1
        irradiance_output_tex[px] = float4(
            irradiance * reservoir.W
            //TODO
             //* saturate(dot(dir_sel, center_normal_ws))
             , 1);

            //irradiance_output_tex[px] = reservoir.w_sum / max(1e-5, calculate_luma(irradiance)) * 0.001;
            //irradiance_output_tex[px] = float4(irradiance, 1);
            //irradiance_output_tex[px] = reservoir.W * 0.01;
        return;
    #endif*/
}
