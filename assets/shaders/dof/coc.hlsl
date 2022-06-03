#include "../inc/frame_constants.hlsl"
#include "../inc/samplers.hlsl"

[[vk::binding(0)]] Texture2D<float> depth_tex;
[[vk::binding(1)]] RWTexture2D<float> output_tex;
[[vk::binding(2)]] RWTexture2D<float> tile_output_tex;

groupshared uint max_abs_coc_asuint;

float coc_size(float depth, float focal_point, float focus_scale) {
    //return 0;
	float coc = clamp((1.0 / focal_point - 1.0 / depth) * focus_scale, -0.3, 0.3);
	return coc * 20.0;
}

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    max_abs_coc_asuint = asuint(0.0);
    GroupMemoryBarrierWithGroupSync();

    float linear_depth = -depth_to_view_z(depth_tex[px]);
    float max_coc = 20.0;

    float focus = -depth_to_view_z(depth_tex.SampleLevel(sampler_nnc, 0.5, 0));
    
    //float coc = clamp((linear_depth - 1) * 20.0, -max_coc, max_coc);
    float coc = coc_size(linear_depth, focus, 0.7);

    InterlockedMax(max_abs_coc_asuint, asuint(abs(coc)));
    GroupMemoryBarrierWithGroupSync();

    if (0 == idx_within_group) {
        tile_output_tex[px / 8] = asfloat(max_abs_coc_asuint);
    }

    output_tex[px] = coc;
}
