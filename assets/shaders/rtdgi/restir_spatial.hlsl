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
#include "rtdgi_restir_settings.hlsl"
#include "rtdgi_common.hlsl"
#include "occlusion_raymarch.hlsl"

[[vk::binding(0)]] Texture2D<uint2> reservoir_input_tex;
[[vk::binding(1)]] Texture2D<float3> bounced_radiance_input_tex;
[[vk::binding(2)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(3)]] Texture2D<float> half_depth_tex;
[[vk::binding(4)]] Texture2D<float> depth_tex;
[[vk::binding(5)]] Texture2D<float> half_ssao_tex;
[[vk::binding(6)]] Texture2D<uint4> temporal_reservoir_packed_tex;
[[vk::binding(7)]] Texture2D<float3> reprojected_gi_tex;
[[vk::binding(8)]] RWTexture2D<uint2> reservoir_output_tex;
[[vk::binding(9)]] RWTexture2D<float3> bounced_radiance_output_tex;
[[vk::binding(10)]] cbuffer _ {
    float4 gbuffer_tex_size;
    float4 output_tex_size;
    uint spatial_reuse_pass_idx;
    // Only done in the last spatial resampling pass
    uint perform_occlusion_raymarch;
    uint occlusion_raymarch_importance_only;
};

#define USE_SSAO_WEIGHING 1
#define ALLOW_REUSE_OF_BACKFACING 1

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

// Two-thirds of SmeLU
float normal_inluence_nonlinearity(float x, float b) {
    return select(x < -b
        , 0
        , (x + b) * (x + b) / (4 * b));
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    const uint2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;
    float depth = half_depth_tex[px];

    const uint seed = frame_constants.frame_index + spatial_reuse_pass_idx * 123;
    uint rng = hash3(uint3(px, seed));

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float3 center_normal_vs = half_view_normal_tex[px].rgb;
    const float3 center_normal_ws = direction_view_to_world(center_normal_vs);
    const float center_depth = half_depth_tex[px];
    const float center_ssao = half_ssao_tex[px].r;

    Reservoir1sppStreamState stream_state = Reservoir1sppStreamState::create();
    Reservoir1spp reservoir = Reservoir1spp::create();

    float3 dir_sel = 1;

    float sample_radius_offset = uint_to_u01_float(hash1_mut(rng));

    Reservoir1spp center_r = Reservoir1spp::from_raw(reservoir_input_tex[px]);

    float kernel_tightness = 1.0 - center_ssao;

    const uint SAMPLE_COUNT_PASS0 = 8;
    const uint SAMPLE_COUNT_PASS1 = 5;

    const float MAX_INPUT_M_IN_PASS0 = RESTIR_TEMPORAL_M_CLAMP;
    const float MAX_INPUT_M_IN_PASS1 = MAX_INPUT_M_IN_PASS0 * SAMPLE_COUNT_PASS0;
    const float MAX_INPUT_M_IN_PASS = select(spatial_reuse_pass_idx == 0, MAX_INPUT_M_IN_PASS0, MAX_INPUT_M_IN_PASS1);

    // TODO: consider keeping high in areas of high variance.
    if (RTDGI_RESTIR_SPATIAL_USE_KERNEL_NARROWING) {
        kernel_tightness = lerp(
            kernel_tightness, 1.0,
            0.5 * smoothstep(MAX_INPUT_M_IN_PASS * 0.5, MAX_INPUT_M_IN_PASS, center_r.M));
    }

    float max_kernel_radius =
        select(spatial_reuse_pass_idx == 0
        , lerp(32.0, 12.0, kernel_tightness)
        , lerp(16.0, 6.0, kernel_tightness));

    // TODO: only run more passes where absolutely necessary (dispatch in tiles)
    if (spatial_reuse_pass_idx >= 2) {
        max_kernel_radius = 8;
    }

    const float2 dist_to_edge_xy = min(float2(px), output_tex_size.xy - px);
    const float allow_edge_overstep = select(center_r.M < 10, 100.0, 1.25);
    //const float allow_edge_overstep = 1.25;
    const float2 kernel_radius = min(max_kernel_radius, dist_to_edge_xy * allow_edge_overstep);
    //const float2 kernel_radius = max_kernel_radius;

    uint sample_count = select(DIFFUSE_GI_USE_RESTIR
        , select(spatial_reuse_pass_idx == 0, SAMPLE_COUNT_PASS0, SAMPLE_COUNT_PASS1)
        , 1);

    #if 1
        // Scrambling angles here would be nice, but results in bad cache thrashing.
        // Quantizing the offsets results in mild cache abuse, and fixes most of the artifacts
        // (flickering near edges, e.g. under sofa in the UE5 archviz apartment scene).
        const uint2 ang_offset_seed = select(spatial_reuse_pass_idx == 0
            , (px >> 3)
            , (px >> 2));
    #else
        // Haha, cache go brrrrrrr.
        const uint2 ang_offset_seed = px;
    #endif

    float ang_offset = uint_to_u01_float(hash3(
        uint3(ang_offset_seed, frame_constants.frame_index * 2 + spatial_reuse_pass_idx)
    )) * M_PI * 2;

    if (!RESTIR_USE_SPATIAL) {
        sample_count = 1;
    }

    float3 radiance_output = 0;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        //float ang = M_PI / 2;
        float ang = (sample_i + ang_offset) * GOLDEN_ANGLE;
        float2 radius =
            select(0 == sample_i
            , 0
            , (pow(float(sample_i + sample_radius_offset) / sample_count, 0.5) * kernel_radius));
        int2 rpx_offset = float2(cos(ang), sin(ang)) * radius;

        const bool is_center_sample = sample_i == 0;
        //const bool is_center_sample = all(rpx_offset == 0);

        const int2 rpx = px + rpx_offset;

        const uint2 reservoir_raw = reservoir_input_tex[rpx];
        if (0 == reservoir_raw.x) {
            // Invalid reprojectoin
            continue;
        }

        Reservoir1spp r = Reservoir1spp::from_raw(reservoir_raw);

        r.M = min(r.M, 500);

        const uint2 spx = reservoir_payload_to_px(r.payload);

        const TemporalReservoirOutput spx_packed = TemporalReservoirOutput::from_raw(temporal_reservoir_packed_tex[spx]);
        const float reused_luminance = spx_packed.luminance;

        float visibility = 1;
        float relevance = 1;

        // Note: we're using `rpx` (neighbor reservoir px) here instead of `spx` (original ray px),
        // since we're merging with the stream of the neighbor and not the original ray.
        //
        // The distinction is in jacobians -- during every exchange, they get adjusted so that the target
        // pixel has correctly distributed rays. If we were to merge with the original pixel's stream,
        // we'd be applying the reservoirs several times.
        //
        // Consider for example merging a pixel with itself (no offset) multiple times over; we want
        // the jacobian to be 1.0 in that case, and not to reflect wherever its ray originally came from.

        const int2 sample_offset = int2(px) - int2(rpx);
        const float sample_dist2 = dot(sample_offset, sample_offset);
        const float3 sample_normal_vs = half_view_normal_tex[rpx].rgb;

        float3 sample_radiance;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            sample_radiance = bounced_radiance_input_tex[rpx];
        }

        const float normal_similarity_dot = dot(sample_normal_vs, center_normal_vs);
        #if ALLOW_REUSE_OF_BACKFACING
            // Allow reuse even with surfaces that face away, but weigh them down.
            relevance *= normal_inluence_nonlinearity(normal_similarity_dot, 0.5)
                / normal_inluence_nonlinearity(1.0, 0.5);
        #else
            relevance *= max(0, normal_similarity_dot);
        #endif

        const float sample_ssao = half_ssao_tex[rpx];

        #if USE_SSAO_WEIGHING
            relevance *= 1 - abs(sample_ssao - center_ssao);
        #endif

        const float2 rpx_uv = get_uv(
            rpx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            gbuffer_tex_size);
        const float rpx_depth = half_depth_tex[rpx];
        
        if (rpx_depth == 0.0) {
            continue;
        }

        const ViewRayContext rpx_ray_ctx = ViewRayContext::from_uv_and_depth(rpx_uv, rpx_depth);

        const float2 spx_uv = get_uv(
            spx * 2 + HALFRES_SUBSAMPLE_OFFSET,
            gbuffer_tex_size);
        const ViewRayContext spx_ray_ctx = ViewRayContext::from_uv_and_depth(spx_uv, spx_packed.depth);
        const float3 sample_hit_ws = spx_packed.ray_hit_offset_ws + spx_ray_ctx.ray_hit_ws();

        const float3 reused_dir_to_sample_hit_unnorm_ws = sample_hit_ws - rpx_ray_ctx.ray_hit_ws();

        //const float reused_luminance = sample_hit_ws_and_luminance.a;

        // Note: we want the neighbor's sample, which might have been resampled already.
        const float reused_dist = length(reused_dir_to_sample_hit_unnorm_ws);
        const float3 reused_dir_to_sample_hit_ws = reused_dir_to_sample_hit_unnorm_ws / reused_dist;

        const float3 dir_to_sample_hit_unnorm = sample_hit_ws - view_ray_context.ray_hit_ws();
        const float dist_to_sample_hit = length(dir_to_sample_hit_unnorm);
        const float3 dir_to_sample_hit = normalize(dir_to_sample_hit_unnorm);

        // Reject neighbors with vastly different depths
        if (!is_center_sample) {
            // Clamp the normal_vs.z so that we don't get arbitrarily loose depth comparison at grazing angles.
            const float depth_diff = abs(max(0.3, center_normal_vs.z) * (center_depth / rpx_depth - 1.0));

            const float depth_threshold =
                select(spatial_reuse_pass_idx == 0
                , 0.15
                , 0.1);

            relevance *= 1 - smoothstep(0.0, depth_threshold, depth_diff);
        }

        // Raymarch to check occlusion
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH && perform_occlusion_raymarch) {
            const float2 ray_orig_uv = spx_uv;

        	//const float surface_offset_len = length(spx_ray_ctx.ray_hit_vs() - view_ray_context.ray_hit_vs());
            const float surface_offset_len = length(
                // Use the center depth for simplicity; this doesn't need to be exact.
                // Faster, looks about the same.
                ViewRayContext::from_uv_and_depth(ray_orig_uv, depth).ray_hit_vs() - view_ray_context.ray_hit_vs()
            );

            // Multiplier over the surface offset from the center to the neighbor
            const float MAX_RAYMARCH_DIST_MULT = 3.0;

            // Trace towards the hit point.

            const float3 raymarch_dir_unnorm_ws = sample_hit_ws - view_ray_context.ray_hit_ws();
            const float3 raymarch_end_ws =
                view_ray_context.ray_hit_ws()
                // TODO: what's a good max distance to raymarch?
                + raymarch_dir_unnorm_ws * min(1.0, MAX_RAYMARCH_DIST_MULT * surface_offset_len / length(raymarch_dir_unnorm_ws));

            OcclusionScreenRayMarch raymarch = OcclusionScreenRayMarch::create(
                uv, view_ray_context.ray_hit_cs.xyz, view_ray_context.ray_hit_ws(),
                raymarch_end_ws,
                gbuffer_tex_size.xy
            )
            .with_max_sample_count(6)
            .with_halfres_depth(output_tex_size.xy, half_depth_tex);
            //.with_fullres_depth(depth_tex);

            if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                raymarch = raymarch.with_color_bounce(reprojected_gi_tex);
            }
            
            raymarch.march(visibility, sample_radiance);
		}

        const float3 sample_hit_normal_ws = spx_packed.hit_normal_ws;

        // phi_2^r in the ReSTIR GI paper
        const float center_to_hit_vis = -dot(sample_hit_normal_ws, dir_to_sample_hit);

        // phi_2^q
        const float reused_to_hit_vis = -dot(sample_hit_normal_ws, reused_dir_to_sample_hit_ws);

        float p_q = 1;
        if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
            p_q *= sRGB_to_luminance(sample_radiance);
        } else {
            p_q *= reused_luminance;
        }

        // Unlike in temporal reuse, here we can (and should) be running this.
        p_q *= max(0, dot(dir_to_sample_hit, center_normal_ws));

        float jacobian = 1;

        // Distance falloff. Needed to avoid leaks.
        jacobian *= reused_dist / dist_to_sample_hit;
        jacobian *= jacobian;

        // N of hit dot -L. Needed to avoid leaks. Without it, light "hugs" corners.
        //
        // Note: importantly, using the neighbor's data, not the original ray.
        jacobian *= clamp(center_to_hit_vis / reused_to_hit_vis, 0, 1e4);

        // Clearly wrong, but!:
        // The Jacobian introduces additional noise in corners, which is difficult to filter.
        // We still need something _resembling_ the jacobian in order to get directional cutoff,
        // and avoid leaks behind surfaces, but we don't actually need the precise Jacobian.
        // This causes us to lose some energy very close to corners, but with the near field split,
        // we don't need it anyway -- and it's better not to have the larger dark halos near corners,
        // which fhe full jacobian can cause due to imperfect integration (color bbox filters, etc).
        jacobian = sqrt(jacobian);

        if (is_center_sample) {
            jacobian = 1;
        }

        // Clamp neighbors give us a hit point that's considerably easier to sample
        // from our own position than from the neighbor. This can cause some darkening,
        // but prevents fireflies.
        //
        // The darkening occurs in corners, where micro-bounce should be happening instead.

        if (RTDGI_RESTIR_USE_JACOBIAN_BASED_REJECTION) {
            #if 1
                // Doesn't over-darken corners as much
                jacobian = min(jacobian, RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE);
            #else
                // Slightly less noise
                if (jacobian > RTDGI_RESTIR_JACOBIAN_BASED_REJECTION_VALUE) { continue; }
            #endif
        }

        if (!(p_q >= 0)) {
            continue;
        }

        r.M *= relevance;

        if (occlusion_raymarch_importance_only) {
            // This is used with ray-traced reservoir visibility which happens after
            // the last spatial resampling. We don't _need_ to perform the raymarch
            // for it, but importance sampling based on unshadowed contribution
            // could end up choosing occluded areas, which then get turned black
            // by the ray-traced check. This then creates extra variance.
            //
            // We can instead try to use the ray-marched visibility as an estimator
            // of real visibility.

            p_q *= lerp(0.25, 1.0, visibility);
            visibility = 1;
        }

        if (reservoir.update_with_stream(
            r, p_q, visibility * jacobian,
            stream_state, r.payload, rng
        )) {
            dir_sel = dir_to_sample_hit;
            radiance_output = sample_radiance;
        }
    }

    reservoir.finish_stream(stream_state);
    reservoir.W = min(reservoir.W, RESTIR_RESERVOIR_W_CLAMP);

    reservoir_output_tex[px] = reservoir.as_raw();

    if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
        bounced_radiance_output_tex[px] = radiance_output;
    }
}
