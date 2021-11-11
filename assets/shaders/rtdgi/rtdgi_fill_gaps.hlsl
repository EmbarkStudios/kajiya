#include "../inc/math.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    const float4 center = input_tex[px];
    if (true || center.w > 0) {
        output_tex[px] = center;
        return;
    }

    const float center_depth = depth_tex[px];

    float4 sum = center;
    float w_sum = center.w > 0 ? 1 : 0;

    // TODO: do something that destroys caches less.

    const float ang_off = (frame_constants.frame_index * 3) % 32 * M_PI * 2;
    
    [loop]
    for (int sample_i = 0; sample_i < 32 && sum.w < 16; ++sample_i) {
        float ang = (ang_off + sample_i + 1) * GOLDEN_ANGLE;
        float radius = sample_i < 4
            ? (sample_i + 1) * 1.5
            : (5 * 1.5 + (sample_i - 4) * 4.5);
        //float radius = (sample_i + 1) * 4.5;
        //float radius = pow(float(sample_i + 1), 1.5) * 1.5;
        int2 sample_offset = float2(cos(ang), sin(ang)) * radius;
        const int2 spx = int2(px) + sample_offset;

        float4 s_contrib = input_tex[spx];
        if (s_contrib.w > 0) {
            // Clamp, so we blur more samples than one
            s_contrib.w = min(s_contrib.w, 4);

            float s_depth = depth_tex[spx];

            if (inverse_depth_relative_diff(center_depth, s_depth) < 0.1 * sqrt(1 + sample_i)) {
                const float w = 1;
                sum += s_contrib * w;
                w_sum += w;
            }
        }
    }

    float4 result = float4(sum.xyz / max(1, w_sum), sum.w);
    output_tex[px] = result;
}
