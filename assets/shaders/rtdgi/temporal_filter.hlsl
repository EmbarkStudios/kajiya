#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/soft_color_clamp.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/gbuffer.hlsl"

#include "../inc/working_color_space.hlsl"
#define linear_to_working linear_rgb_to_linear_luma_chroma
#define working_to_linear linear_luma_chroma_to_linear_rgb

#define USE_TEMPORAL_FILTER 1

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(4)]] Texture2D<float> half_depth_tex;
[[vk::binding(5)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(6)]] RWTexture2D<float4> history_output_tex;
[[vk::binding(7)]] RWTexture2D<float4> output_tex;
[[vk::binding(8)]] cbuffer _ {
    float4 output_tex_size;
    float4 gbuffer_tex_size;
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    const float4 center = linear_to_working(input_tex[px]);
    float4 reproj = reprojection_tex[px * 2];
    float4 history = linear_to_working(history_tex.SampleLevel(sampler_lnc, uv + reproj.xy, 0));
    
#if 1
	float4 vsum = 0.0.xxxx;
	float4 vsum2 = 0.0.xxxx;
	float wsum = 0.0;

	const int k = 2;
    {for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = linear_to_working(input_tex[px + int2(x, y)]);
			float w = 1;//exp(-3.0 * float(x * x + y * y) / float((k+1.) * (k+1.)));
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
#else
	float4 nmin = center;
	float4 nmax = center;

	const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float4 neigh = linear_to_working(input_tex[px + int2(x, y) * 2]);
			nmin = min(nmin, neigh);
            nmax = max(nmax, neigh);
        }
    }
#endif

    const float light_stability = 1;

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

    // haaaaaaax
    // TODO: do a better disocclussion check
    // reproj_validity_dilated = 1;

#if 0
	float4 clamped_history = clamp(history, nmin, nmax);
#else
    float4 clamped_history = float4(
        soft_color_clamp(center.rgb, history.rgb, ex.rgb, 0.75 * dev.rgb),
        history.a
    );
#endif
    //clamped_history = center;
    //clamped_history = history;

    // TODO: proper rejection (not "reproj_validity_dilated")
    float3 res = lerp(clamped_history.rgb, center.rgb, 1.0 / lerp(1.0, 16.0, reproj_validity_dilated * light_stability));

    #if !USE_TEMPORAL_FILTER
        res = center.rgb;
    #endif

    float4 spatial_input = working_to_linear(float4(res, 0));

    history_output_tex[px] = spatial_input;

    //spatial_input *= reproj.z;    // debug validity
    //spatial_input *= light_stability;
    //spatial_input = length(dev.rgb);
    //spatial_input = 1-light_stability;
    //spatial_input = abs(dev.rgb);
    output_tex[px] = float4(spatial_input.rgb, light_stability);
    //history_output_tex[px] = reproj.w;
}
