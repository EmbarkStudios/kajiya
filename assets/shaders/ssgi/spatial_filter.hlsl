#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"

[[vk::binding(0)]] Texture2D<float4> ssgi_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> normal_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;

#define USE_AO_ONLY 1

float4 process_sample(float4 ssgi, float depth, float3 normal, float center_depth, float3 center_normal, inout float w_sum) {
    if (depth != 0.0)
    {
        //float depth_diff = (1.0 / center_depth) - (1.0 / depth);
        //float depth_factor = exp2(-(200.0 * center_depth) * abs(depth_diff));
        float depth_diff = 1.0 - (center_depth / depth);
        float depth_factor = exp2(-200.0 * abs(depth_diff));

        float normal_factor = max(0.0, dot(normal, center_normal));
        normal_factor *= normal_factor;
        normal_factor *= normal_factor;

        float w = 1;
        w *= depth_factor;  // TODO: differentials
        w *= normal_factor;

        w_sum += w;
        return ssgi * w;
    } else {
        return 0.0.xxxx;
    }
}

[numthreads(8, 8, 1)]
void main(uint2 px : SV_DispatchThreadID) {
    #if 0
        output_tex[px] = ssgi_tex[px];
        return;
    #endif

    float4 result = 0.0.xxxx;
    float w_sum = 0.0;

    float center_depth = depth_tex[px].x;
    if (center_depth != 0.0) {
        float3 center_normal = normal_tex[px].xyz;

    	float4 center_ssgi = ssgi_tex[px];
        w_sum = 1.0;
        result = center_ssgi;

        const int kernel_half_size = 1;
        for (int y = -kernel_half_size; y <= kernel_half_size; ++y) {
            for (int x = -kernel_half_size; x <= kernel_half_size; ++x) {
                if (x != 0 || y != 0) {
                    int2 sample_px = px + int2(x, y);
                    float depth = depth_tex[sample_px].x;
                    float4 ssgi = ssgi_tex[sample_px];
                    float3 normal = normal_tex[sample_px].xyz;
                    result += process_sample(ssgi, depth, normal, center_depth, center_normal, w_sum);
                }
            }
        }
    } else {
        result = 0.0.xxxx;
    }

    #if USE_AO_ONLY
        result = result.r;
    #endif

	output_tex[px] = result / max(w_sum, 1e-5);
}
