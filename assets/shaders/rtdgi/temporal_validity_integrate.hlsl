#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/math.hlsl"
#include "rtdgi_restir_settings.hlsl"

#define USE_SSAO_STEERING 1
#define USE_DYNAMIC_KERNEL_RADIUS 0

[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] Texture2D<float> history_tex;
[[vk::binding(2)]] Texture2D<float4> reprojection_tex;
[[vk::binding(3)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(4)]] Texture2D<float> half_depth_tex;
[[vk::binding(5)]] RWTexture2D<float2> output_tex;
[[vk::binding(6)]] cbuffer _ {
    float4 gbuffer_tex_size;
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 invalid_blurred = 0;

    const float center_depth = half_depth_tex[px];
    const float4 reproj = reprojection_tex[px * 2];

    if (RESTIR_USE_PATH_VALIDATION) {
    	{const int k = 2;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                const int2 offset = int2(x, y);
                //float w = 1;
                float w = exp2(-0.1 * dot(offset, offset));
                invalid_blurred += float2(input_tex[px + offset], 1) * w;
            }
        }}
        invalid_blurred /= invalid_blurred.y;
        invalid_blurred.x = lerp(invalid_blurred.x, WaveReadLaneAt(invalid_blurred.x, WaveGetLaneIndex() ^ 2), 0.5);
        invalid_blurred.x = lerp(invalid_blurred.x, WaveReadLaneAt(invalid_blurred.x, WaveGetLaneIndex() ^ 16), 0.5);
        //invalid_blurred.x = lerp(invalid_blurred.x, WaveActiveSum(invalid_blurred.x) / 64.0, 0.25);

        const float2 reproj_rand_offset = 0.0;

        invalid_blurred.x = smoothstep(0.0, 1.0, invalid_blurred.x);
    }
    
    /*if (reproj.z == 0) {
        invalid_blurred.x += 1;
    }*/

    float edge = 1;

	{const int k = 2;
    for (int y = 0; y <= k; ++y) {
        for (int x = 1; x <= k; ++x) {
            const int2 offset = int2(x, y);
            const int2 sample_px = px * 2 + offset;
            const int2 sample_px_half = px + offset / 2;
            const float4 reproj = reprojection_tex[sample_px];
            const float sample_depth = half_depth_tex[sample_px_half];

            if (reproj.w < 0 || inverse_depth_relative_diff(center_depth, sample_depth) > 0.1) {
                edge = 0;
                break;
            }

            edge *= reproj.z == 0 && sample_depth != 0;
        }
    }}

    edge = max(edge, WaveReadLaneAt(edge, WaveGetLaneIndex() ^ 1));
    edge = max(edge, WaveReadLaneAt(edge, WaveGetLaneIndex() ^ 8));
    /*edge = max(edge, WaveReadLaneAt(edge, WaveGetLaneIndex() ^ 4));
    edge = max(edge, WaveReadLaneAt(edge, WaveGetLaneIndex() ^ 32));*/

    invalid_blurred.x += edge;

    invalid_blurred = saturate(invalid_blurred);

    //invalid_blurred.x = smoothstep(0.1, 1.0, invalid_blurred.x);
    //invalid_blurred.x = pow(invalid_blurred.x, 4);


    //invalid_blurred.x = 1;;

    const float2 reproj_px = px + gbuffer_tex_size.xy * reproj.xy / 2 + 0.5;
    float history = 0;

    const int sample_count = 8;
    float ang_off = uint_to_u01_float(hash3(uint3(px, frame_constants.frame_index))) * M_PI * 2;

    for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
        float ang = (sample_i + ang_off) * GOLDEN_ANGLE;
        float radius = float(sample_i) * 1.0;
        float2 sample_offset = float2(cos(ang), sin(ang)) * radius;
        const int2 sample_px = int2(reproj_px + sample_offset);
        history += history_tex[sample_px];
    }

    history /= sample_count;


    /*float history = (
        history_tex[reproj_px] +
        history_tex[reproj_px + int2(-4, 0)] +
        history_tex[reproj_px + int2(4, 0)] +
        history_tex[reproj_px + int2(0, 4)] +
        history_tex[reproj_px + int2(0, -4)]
    ) / 5;*/

    //float history = history_tex[reproj_px];

    output_tex[px] = float2(
        max(history * 0.75, invalid_blurred.x),
        //invalid_blurred.x,
        input_tex[px]
    );
}
