#include "../inc/pack_unpack.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/gbuffer.hlsl"

[[vk::binding(0)]] Texture2D<float4> ssgi_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;

#define USE_AO_ONLY 1

float4 process_sample(float2 soffset, float4 ssgi, float depth, float3 normal, float center_depth, float3 center_normal, inout float w_sum) {
    if (depth != 0.0)
    {
        float depth_diff = 1.0 - (center_depth / depth);
        float depth_factor = exp2(-200.0 * abs(depth_diff));

        float normal_factor = max(0.0, dot(normal, center_normal));
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;

        float w = 1;
        w *= depth_factor;  // TODO: differentials
        w *= normal_factor;
        w *= exp(-dot(soffset, soffset));

        w_sum += w;
        return ssgi * w;
    } else {
        return 0.0.xxxx;
    }
}

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {

    float4 result = 0.0.xxxx;
    float w_sum = 0.0;

    float center_depth = depth_tex[px];
    if (center_depth != 0.0) {
        float3 center_normal = unpack_normal_11_10_11(gbuffer_tex[px].y);

    	float4 center_ssgi = 0.0.xxxx;
        w_sum = 0.0;
        result = center_ssgi;

        const int kernel_half_size = 1;
        for (int y = -kernel_half_size; y <= kernel_half_size; ++y) {
            for (int x = -kernel_half_size; x <= kernel_half_size; ++x) {
                int2 sample_pix = px / 2 + int2(x, y) * 1;
                float depth = depth_tex[sample_pix * 2];
                float4 ssgi = ssgi_tex[sample_pix];
                float3 normal = unpack_normal_11_10_11(gbuffer_tex[sample_pix * 2].y);
                result += process_sample(float2(x, y), ssgi, depth, normal, center_depth, center_normal, w_sum);
            }
        }
    } else {
        result = 0.0.xxxx;
    }

    #if USE_AO_ONLY
        result = result.r;
    #endif

    if (w_sum > 1e-6) {
        output_tex[px] = result / w_sum;
    } else {
        output_tex[px] = ssgi_tex[px / 2];
    }
}
