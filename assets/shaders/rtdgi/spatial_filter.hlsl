#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"

[[vk::binding(0)]] Texture2D<float4> hit0_tex;
[[vk::binding(1)]] Texture2D<float4> hit1_tex;
[[vk::binding(2)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(3)]] Texture2D<float> half_depth_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float4 sum = 0;

    float3 center_normal_vs = half_view_normal_tex[px].rgb;
    float depth = half_depth_tex[px];
    float3 center_val = hit0_tex[px].rgb;

    int k = 3;
    int skip = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const int2 sample_offset = int2(x, y) * skip;
            const int2 sample_px = px + sample_offset;

            float3 sample_normal_vs = half_view_normal_tex[sample_px].rgb;
            float sample_depth = half_depth_tex[sample_px];
            float3 sample_val = hit0_tex[sample_px].rgb;

            if (sample_depth != 0) {
                float wt = exp2(-0.4 * sqrt(float(dot(sample_offset, sample_offset))));
                wt *= pow(saturate(dot(center_normal_vs, sample_normal_vs)), 4);
                wt *= exp2(-15.0 * abs(depth / sample_depth - 1.0));

                // Preserve edges in value
                wt *= exp2(-8.0 * abs(calculate_luma(sample_val - center_val)));

                sum += float4(sample_val, 1) * wt;
            }
        }
    }

    output_tex[px] = float4(sum.rgb / max(1e-5, sum.a), 1.0);
}
