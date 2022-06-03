#include "../inc/color.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"

#define USE_SSAO_STEERING 1
#define USE_DYNAMIC_KERNEL_RADIUS 0

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> ssao_tex;
[[vk::binding(3)]] Texture2D<float3> geometric_normal_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
};

float square(float x) { return x * x; }
float max3(float x, float y, float z) { return max(x, max(y, z)); }

// Bias towards dimmer input -- we don't care about energy loss here
// since this does not feed into subsequent passes, but want to minimize noise.
//
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
float3 crunch(float3 v) {
    return v * rcp(max3(v.r, v.g, v.b) + 1.0);
}
float3 uncrunch(float3 v) {
    return v * rcp(1.0 - max3(v.r, v.g, v.b));
}

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    #if 0
        output_tex[px] = input_tex[px];
        return;
    #endif
    
    float4 sum = 0;

    const float center_validity = input_tex[px].a;
    const float center_depth = depth_tex[px];
    const float center_ssao = ssao_tex[px].r;
    const float3 center_value = input_tex[px].rgb;
    const float3 center_normal_vs = geometric_normal_tex[px] * 2.0 - 1.0;

    if (center_validity == 1) {
        output_tex[px] = float4(center_value, 1.0);
        return;
    }

    const float ang_off = (frame_constants.frame_index * 23) % 32 * M_TAU + interleaved_gradient_noise(px) * M_PI;

    const uint MAX_SAMPLE_COUNT = 8;
    const float MAX_RADIUS_PX = sqrt(lerp(16.0 * 16.0, 2.0 * 2.0, center_validity));

    // Feeds into the `pow` to remap sample index to radius.
    // At 0.5 (sqrt), it's proper circle sampling, with higher values becoming conical.
    // Must be constants, so the `pow` can be const-folded.
    const float KERNEL_SHARPNESS = 0.666;

    const uint sample_count = clamp(uint(exp2(4.0 * square(1.0 - center_validity))), 2, MAX_SAMPLE_COUNT);

    {
        sum += float4(crunch(center_value), 1);

        const float RADIUS_SAMPLE_MULT = MAX_RADIUS_PX / pow(float(MAX_SAMPLE_COUNT - 1), KERNEL_SHARPNESS);

        // Note: faster on RTX2080 than a dynamic loop
        for (uint sample_i = 1; sample_i < MAX_SAMPLE_COUNT; ++sample_i) {
            const float ang = (sample_i + ang_off) * GOLDEN_ANGLE;

            float radius = pow(float(sample_i), KERNEL_SHARPNESS) * RADIUS_SAMPLE_MULT;
            float2 sample_offset = float2(cos(ang), sin(ang)) * radius;
            const int2 sample_px = px + sample_offset;

            const float sample_depth = depth_tex[sample_px];
            const float3 sample_val = input_tex[sample_px].rgb;
            const float sample_ssao = ssao_tex[sample_px].r;
            const float3 sample_normal_vs = geometric_normal_tex[sample_px] * 2.0 - 1.0;

            if (sample_depth != 0 && sample_i < sample_count) {
                float wt = 1;
                //wt *= pow(saturate(dot(center_normal_vs, sample_normal_vs)), 20);
                wt *= exp2(-100.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));

                #if USE_SSAO_STEERING
                    wt *= exp2(-20.0 * abs(sample_ssao - center_ssao));
                #endif

                sum += float4(crunch(sample_val), 1.0) * wt;
            }
        }
    }

    float norm_factor = 1.0 / max(1e-5, sum.a);
    float3 filtered = uncrunch(sum.rgb * norm_factor);

    output_tex[px] = float4(filtered, 1.0);
}
