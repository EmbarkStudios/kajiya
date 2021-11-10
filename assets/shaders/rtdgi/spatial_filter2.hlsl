#include "../inc/color.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"

#define USE_SSAO_STEERING 1
#define USE_DYNAMIC_KERNEL_RADIUS 0

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> ssao_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    #if 0
        output_tex[px] = input_tex[px];
        return;
    #endif
    
    float4 sum = 0;
    float ex = 0;
    float ex2 = 0;

    const float center_validity = input_tex[px].a;
    const float center_depth = depth_tex[px];
    const float center_ssao = ssao_tex[px].r;

    #define USE_POISSON 1

    //float4 blue = blue_noise_for_pixel(px, frame_constants.frame_index);
    //float4 blue = blue_noise_for_pixel(px, frame_constants.frame_index);
    float2 blue = r2_sequence(frame_constants.frame_index % 128);

    float spatial_sharpness = 0.25;//lerp(0.5, 0.25, saturate(center_ssao));
    //const int sample_count = max(1, 16 * saturate(1.0 - center_validity));
    const int sample_count = clamp(int(exp2(4 * saturate(1.0 - center_validity))), 1, 16);
    //const int sample_count = 16;

    {
        for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
            float ang = (sample_i + blue.x) * GOLDEN_ANGLE;
            //float radius = 1.5 + float(sample_i) * lerp(0.333, 0.8, center_ssao);
            float radius = sqrt(float(sample_i)) * 4.0;
            int2 sample_offset = float2(cos(ang), sin(ang)) * radius;
            const int2 sample_px = px + sample_offset;

            const float sample_depth = depth_tex[sample_px];
            const float3 sample_val = input_tex[sample_px].rgb;
            const float sample_ssao = ssao_tex[sample_px].r;

            if (sample_depth != 0) {
                float wt = 1;
                //wt *= exp2(-spatial_sharpness * sqrt(float(dot(sample_offset, sample_offset))));
                //wt *= pow(saturate(dot(center_normal_vs, sample_normal_vs)), 20);
                wt *= exp2(-100.0 * abs(/*center_normal_vs.z * */(center_depth / sample_depth - 1.0)));

                #if USE_SSAO_STEERING
                    wt *= exp2(-20.0 * abs(sample_ssao - center_ssao));
                #endif

                sum += float4(sample_val, 1) * wt;
                
                float luma = calculate_luma(sample_val);
                ex += luma * wt;
                ex2 += luma * luma * wt;
            }

            // Adaptive stopping
            /*if (sample_i >= 3) {
                float var = abs(ex2 / sum.a - (ex / sum.a) * (ex / sum.a));
                var *= (sample_i + 1) / sample_i;   // Bessel's correction, 0-based
                float rel_dev = sqrt(var) / (abs(ex) / sum.a);
                if (rel_dev < 0.3)
                {
                    break;
                }
            }*/
        }
    }

    float norm_factor = 1.0 / max(1e-5, sum.a);
    float3 filtered = sum.rgb * norm_factor;

    output_tex[px] = float4(filtered, 1.0);
}
