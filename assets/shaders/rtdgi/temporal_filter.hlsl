#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> cv_history_tex;
[[vk::binding(3)]] Texture2D<float2> variance_history_tex;
[[vk::binding(4)]] Texture2D<float4> reprojection_tex;
[[vk::binding(5)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(6)]] Texture2D<float> half_depth_tex;
[[vk::binding(7)]] Texture3D<float4> csgi_direct_tex;
[[vk::binding(8)]] Texture3D<float4> csgi_indirect_tex;
[[vk::binding(9)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(10)]] RWTexture2D<float4> cv_history_output_tex;
[[vk::binding(11)]] RWTexture2D<float2> variance_history_output_tex;
[[vk::binding(12)]] RWTexture2D<float4> output_tex;
[[vk::binding(13)]] cbuffer _ {
    float4 output_tex_size;
    float4 gbuffer_tex_size;
};

#include "../csgi/common.hlsl"
#include "../csgi/lookup.hlsl"


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = input_tex[px];
    float4 reproj = reprojection_tex[px * 2];
    float4 history = history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    
#if 1
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 2;
    {for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = (input_tex[px + int2(x, y) * 2]);
			float w = 1;//exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;
        }
    }}

	float4 ex = vsum / wsum;
	float4 ex2 = vsum2 / wsum;
	float4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    float box_size = 1;//lerp(reproj.w, 1.0, 0.5);

    const float n_deviations = 5.0;
	float4 nmin = lerp(center, ex, box_size * box_size) - dev * box_size * n_deviations;
	float4 nmax = lerp(center, ex, box_size * box_size) + dev * box_size * n_deviations;
#else
	float4 nmin = center;
	float4 nmax = center;

	const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = (input_tex[px + int2(x, y) * 2]);
			nmin = min(nmin, neigh);
            nmax = max(nmax, neigh);
        }
    }
#endif

    float3 control_variate = 0.0.xxx;
    {
        const uint2 hi_px = px * 2;
        const float2 uv = get_uv(hi_px, gbuffer_tex_size);
        float depth = half_depth_tex[px];
        const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

        const float3 ray_hit_ws = view_ray_context.ray_hit_ws();
        const float3 ray_hit_vs = view_ray_context.ray_hit_vs();

        float3 normal = mul(frame_constants.view_constants.view_to_world, float4(half_view_normal_tex[px].rgb, 0)).xyz;

        // TODO: this could use bent normals to avoid leaks, or could be integrated into the SSAO loop,
        // Note: point-lookup doesn't leak, so multiple bounces should be fine
        float3 to_eye = get_eye_position() - ray_hit_ws;
        float3 pseudo_bent_normal = normalize(normalize(to_eye) + normal);

        control_variate = lookup_csgi(
            ray_hit_ws,
            normal,
            CsgiLookupParams::make_default()
                .with_bent_normal(pseudo_bent_normal)
        );
    }
    const float control_variate_luma = calculate_luma(control_variate);

    float history_dist = 1e5; {
        int2 history_px = int2((uv + reproj.xy) * output_tex_size.xy);
        const int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float4 history = history_tex[history_px + int2(x, y)];
                //history_dist = min(history_dist, abs(control_variate_luma - history.a));
                //float dist = abs(control_variate_luma - history.a);
                float dist = abs(control_variate_luma - history.a) / max(1e-5, control_variate_luma + history.a);
                history_dist = min(history_dist, dist);
            }
        }
    }
    //history_dist = WaveActiveMin(history_dist);

    const float4 cv_history_dev_packed = cv_history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    const float3 cv_history = cv_history_dev_packed.rgb;
    const float dev_history = cv_history_dev_packed.a;

    history_dist = min(history_dist, WaveReadLaneAt(history_dist, WaveGetLaneIndex() ^ 1));
    history_dist = min(history_dist, WaveReadLaneAt(history_dist, WaveGetLaneIndex() ^ 8));

    //history_dist = abs(control_variate_luma - calculate_luma(cv_history));

    //const float invalid = smoothstep(0.0, 10.0, history_dist / max(1e-5, min(history.a, control_variate_luma)));
    
    const float light_stability = 1.0 - 0.8 * smoothstep(0.1, 0.5, history_dist);
    //const float light_stability = 1.0 - step(0.01, history_dist);
    //const float light_stability = 1;

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

	float4 clamped_history = clamp(history, nmin, nmax);
    //clamped_history = center;
    //float4 clamped_history = history;

    float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / lerp(1.0, 4.0, reproj_validity_dilated * light_stability));

    const float smoothed_dev = lerp(dev_history, calculate_luma(abs(dev.rgb)), 0.1);

    history_output_tex[px] = float4(res, control_variate_luma);
    cv_history_output_tex[px] = float4(control_variate, smoothed_dev);

    float3 spatial_input;
    if (USE_RTDGI_CONTROL_VARIATES) {
        spatial_input = max(0.0.xxx, res + control_variate);
    } else {
        spatial_input = max(0.0.xxx, res);
    }

    const float2 moments_history = variance_history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0);
    //const float center_luma = calculate_luma(abs(center.rgb));// - 0.5 * calculate_luma(control_variate.rgb));
    const float center_luma = center.y;// - 0.5 * calculate_luma(control_variate.rgb));
    const float2 current_moments = float2(center_luma, center_luma * center_luma);
    variance_history_output_tex[px] = lerp(moments_history, current_moments, 0.1);

    //spatial_input *= reproj.z;    // debug validity
    //spatial_input *= light_stability;
    //spatial_input = smoothstep(0.0, 0.05, history_dist);
    //spatial_input = length(dev.rgb);
    //spatial_input = 1-light_stability;
    //spatial_input = control_variate_luma;
    //spatial_input = abs(cv_diff);
    //spatial_input = abs(dev.rgb);
    //spatial_input = smoothed_dev;

    //const float center_variance = sqrt(max(0.0, moments_history.y - moments_history.x * moments_history.x)) / max(1e-8, abs(moments_history.x));
    const float center_variance = 5 * max(0.0, moments_history.y - moments_history.x * moments_history.x);
    //spatial_input = center_variance * abs(moments_history.x);
    //spatial_input = abs(center_variance);

    // TODO: adaptively sample according to abs(res)
    //spatial_input = max(0.0, abs(res) / max(1e-5, control_variate));

    //output_tex[px] = float4(spatial_input, smoothed_dev * (light_stability > 0.5 ? 1.0 : -1.0));
    output_tex[px] = float4(spatial_input, light_stability);
    //history_output_tex[px] = reproj.w;
}
