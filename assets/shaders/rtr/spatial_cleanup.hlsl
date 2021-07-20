#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float3> geometric_normal_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

#define SHUFFLE_SUBPIXELS 1

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    const float4 center = input_tex[px];
    const float center_depth = depth_tex[px];
    const float center_sample_count = center.w;

    if (center_sample_count >= 16 || center_depth == 0.0) {
        output_tex[px] = center;
        return;
    }

    const float3 center_normal_vs = geometric_normal_tex[px] * 2 - 1;

    // TODO: project the BRDF lobe footprint; this only works for certain roughness ranges
    const float filter_radius_ss = 0.5 * frame_constants.view_constants.view_to_clip[1][1] / -depth_to_view_z(center_depth);
    const uint filter_idx = uint(clamp(filter_radius_ss * 7.0, 0.0, 7.0));

    float3 vsum = 0.0.xxx;
    float wsum = 0.0;

    const uint sample_count = clamp(int(8 - center_sample_count / 2), 4, 16);
    //const uint sample_count = 8;
    const uint kernel_scale = center_sample_count < 4 ? 2 : 1;
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + (SHUFFLE_SUBPIXELS ? 1 : 0) * frame_constants.frame_index) & 3;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        // TODO: precalculate temporal variants
        int2 sample_px = px + kernel_scale * spatial_resolve_offsets[(px_idx_in_quad * 16 + sample_i) + 64 * filter_idx].xy;

        const float3 neigh = input_tex[sample_px].rgb;
        const float sample_depth = depth_tex[sample_px];
        const float3 sample_normal_vs = geometric_normal_tex[sample_px] * 2 - 1;

        float w = 1;
        // TODO: BRDF-based weights
        w *= exp2(-50.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));
        float dp = saturate(dot(center_normal_vs, sample_normal_vs));
        w *= dp * dp * dp;
        
		vsum += neigh * w;
		wsum += w;
    }

    output_tex[px] = float4(vsum / wsum, 1);
}
