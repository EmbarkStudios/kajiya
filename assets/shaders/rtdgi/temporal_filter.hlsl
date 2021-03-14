#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(4)]] Texture2D<float> half_depth_tex;
[[vk::binding(5)]] Texture3D<float4> csgi2_direct_tex;
[[vk::binding(6)]] Texture3D<float4> csgi2_indirect_tex;
[[vk::binding(7)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(8)]] RWTexture2D<float4> output_tex;
[[vk::binding(9)]] cbuffer _ {
    float4 output_tex_size;
    float4 gbuffer_tex_size;
};

#include "../csgi2/common.hlsl"
#include "../csgi2/lookup.hlsl"

//#define LINEAR_TO_WORKING(x) sqrt(x)
//#define WORKING_TO_LINEAR(x) ((x)*(x))

#define LINEAR_TO_WORKING(x) x
#define WORKING_TO_LINEAR(x) x

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = WORKING_TO_LINEAR(input_tex[px]);
    float4 reproj = reprojection_tex[px * 2];
    float4 history = WORKING_TO_LINEAR(history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0));
    
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = WORKING_TO_LINEAR(input_tex[px + int2(x, y) * 2]);
			float w = exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
			vsum += neigh * w;
			vsum2 += neigh * neigh * w;
			wsum += w;
        }
    }

	float4 ex = vsum / wsum;
	float4 ex2 = vsum2 / wsum;
	float4 dev = sqrt(max(0.0.xxxx, ex2 - ex * ex));

    float box_size = lerp(reproj.w, 1.0, 0.5);

    const float n_deviations = 5.0;
	float4 nmin = lerp(center, ex, box_size * box_size) - dev * box_size * n_deviations;
	float4 nmax = lerp(center, ex, box_size * box_size) + dev * box_size * n_deviations;
    
	float4 clamped_history = clamp(history, nmin, nmax);
    float4 res = lerp(clamped_history, center, lerp(1.0, 1.0 / 12.0, reproj.z));

    history_output_tex[px] = LINEAR_TO_WORKING(res);

    float3 spatial_input = 0;
    {
        const uint2 hi_px = px * 2;
        const float2 uv = get_uv(hi_px, gbuffer_tex_size);
        float depth = half_depth_tex[px];
        const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

        const float3 ray_hit_ws = view_ray_context.ray_hit_ws();
        const float3 ray_hit_vs = view_ray_context.ray_hit_vs();

        float3 normal = mul(frame_constants.view_constants.view_to_world, float4(half_view_normal_tex[px].rgb, 0)).xyz;

        spatial_input = res.rgb;

        #if USE_RTDGI_CONTROL_VARIATES
            spatial_input = max(0.0, spatial_input + lookup_csgi2(
                ray_hit_ws,
                normal,
                Csgi2LookupParams::make_default()
            ));
        #endif
    }

    output_tex[px] = float4(spatial_input, 1);
    //history_output_tex[px] = reproj.w;
}
