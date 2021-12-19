#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/soft_color_clamp.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

#include "../csgi/common.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> cv_history_tex;
[[vk::binding(3)]] Texture2D<float4> reprojection_tex;
[[vk::binding(4)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(5)]] Texture2D<float> half_depth_tex;
[[vk::binding(6)]] Texture3D<float4> csgi_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(7)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(8)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(9)]] RWTexture2D<float4> cv_history_output_tex;
[[vk::binding(10)]] RWTexture2D<float4> output_tex;
[[vk::binding(11)]] cbuffer _ {
    float4 output_tex_size;
    float4 gbuffer_tex_size;
};

#include "../csgi/lookup.hlsl"


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    const float4 center = input_tex[px];
    float4 reproj = reprojection_tex[px * 2];
    float4 history = history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 2;
    {for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = (input_tex[px + int2(x, y) * 2]);
			const float w = 1;
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;
        }
    }}

	float4 ex = vsum / wsum;
	float4 ex2 = vsum2 / wsum;
	float4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    float box_size = 1;

    const float n_deviations = 5.0;
	float4 nmin = lerp(center, ex, box_size * box_size) - dev * box_size * n_deviations;
	float4 nmax = lerp(center, ex, box_size * box_size) + dev * box_size * n_deviations;

    float3 control_variate = 0.0.xxx;
    {
        uint2 hi_px_subpixels[4] = {
            uint2(0, 0),
            uint2(1, 1),
            uint2(1, 0),
            uint2(0, 1),
        };

        const uint2 hi_px = px * 2 + hi_px_subpixels[frame_constants.frame_index & 3];
        const float2 uv = get_uv(hi_px, gbuffer_tex_size);
        float depth = half_depth_tex[px];
        const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

        const float3 ray_hit_ws = view_ray_context.ray_hit_ws();
        const float3 ray_hit_vs = view_ray_context.ray_hit_vs();

        float3 normal = direction_view_to_world(half_view_normal_tex[px].rgb);

        // TODO: this could use bent normals to avoid leaks, or could be integrated into the SSAO loop,
        // Note: point-lookup doesn't leak, so multiple bounces should be fine
        float3 to_eye = get_eye_position() - ray_hit_ws;
        float3 pseudo_bent_normal = normalize(normalize(to_eye) + normal);

        control_variate = lookup_csgi(
            ray_hit_ws,
            normal,
            CsgiLookupParams::make_default()
                .with_bent_normal(pseudo_bent_normal)
                //.with_linear_fetch(false)
        );

        // Brute force control variate calculation that matches the one
        // used in trace_diffuse. The other one is an approximation
        #if 0
            control_variate = 0;

            DiffuseBrdf brdf;
            brdf.albedo = 1.0.xxx;
            const float3x3 tangent_to_world = build_orthonormal_basis(normal);

            const int sample_count = 32;
            for (uint i = 0; i < sample_count; ++i) {
                float2 urand = hammersley(i, sample_count);
                BrdfSample brdf_sample = brdf.sample(float3(0, 0, 1), urand);
                float3 ws_dir = mul(tangent_to_world, brdf_sample.wi);

                control_variate += lookup_csgi(
                    ray_hit_ws,
                    normal,
                    CsgiLookupParams::make_default()
                        .with_sample_directional_radiance(ws_dir)
                        .with_bent_normal(pseudo_bent_normal)
                );
            }

            control_variate /= sample_count;
        #endif
    }
    const float control_variate_luma = calculate_luma(control_variate);

    float history_dist = 1e5; {
        int2 history_px = int2((uv + reproj.xy) * output_tex_size.xy);
        const int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float4 history = history_tex[history_px + int2(x, y)];
                float dist = abs(control_variate_luma - history.a) / max(1e-5, control_variate_luma + history.a);
                history_dist = min(history_dist, dist);
            }
        }
    }

    const float4 cv_history_dev_packed = cv_history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    const float3 cv_history = cv_history_dev_packed.rgb;
    const float dev_history = cv_history_dev_packed.a;

    history_dist = min(history_dist, WaveReadLaneAt(history_dist, WaveGetLaneIndex() ^ 1));
    history_dist = min(history_dist, WaveReadLaneAt(history_dist, WaveGetLaneIndex() ^ 8));

    const float light_stability = 1.0 - 0.8 * smoothstep(0.1, 0.5, history_dist);

    const float3 cv_diff = (control_variate - cv_history);

    float reproj_validity_dilated = reproj.z;
    #if 1
        // Greatly reduces temporal bleeding of slowly-moving edges
        // TODO: do this at sampling stage instead of patching up the bilinear result
        {
         	const int k = 1;
            for (int y = -k; y <= k; ++y) {
                for (int x = -k; x <= k; ++x) {
                    reproj_validity_dilated = min(reproj_validity_dilated, reprojection_tex[px * 2 + int2(x, y)].z);
                }
            }
        }
    #endif

    if (USE_RTDGI_CONTROL_VARIATES) {
        // Temporally stabilize the control variates. Due to the low res nature of CSGI,
        // the control variate can flicker, and very blocky. The abrupt change would eventually
        // be recognized by this temporal filter, but variance in the bounding box clamp makes it lag.
        //
        // Some latency is preferrable to flicker. This will pretend the history had the same control variate
        // as the one we're seeing right now, thus instantly adapting the temporal filter to jumps in CV.
        //
        // Note that this would prevent any changes in lighting, except exponential blending here
        // will slowly blend it over time, with speed similar to if control variates weren't used.
        history.rgb -= cv_diff * reproj_validity_dilated * 0.9;
    }

#if 0
	float4 clamped_history = clamp(history, nmin, nmax);
#else
    float4 clamped_history = float4(
        soft_color_clamp(center.rgb, history.rgb, ex.rgb, dev.rgb),
        history.a
    );
#endif
    //clamped_history = center;
    //clamped_history = history;

    // TODO: proper rejection (not "reproj_validity_dilated")
    float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / lerp(1.0, 4.0, reproj_validity_dilated * light_stability));

    const float smoothed_dev = lerp(dev_history, calculate_luma(abs(dev.rgb)), 0.1);

    history_output_tex[px] = float4(res, control_variate_luma);
    cv_history_output_tex[px] = float4(control_variate, smoothed_dev);

    float3 spatial_input;
    if (USE_RTDGI_CONTROL_VARIATES) {
        // Note: must not be clamped to properly temporally integrate.
        // This value could well end up being negative due to control variate noise,
        // but that is fine, as it will be clamped later.
        spatial_input = res + control_variate;
    } else {
        spatial_input = max(0.0.xxx, res);
    }

    output_tex[px] = float4(spatial_input, light_stability);
}
