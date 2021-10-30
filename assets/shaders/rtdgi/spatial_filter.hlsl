#include "../inc/color.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/hash.hlsl"

#define USE_SSAO_STEERING 1
#define USE_DYNAMIC_KERNEL_RADIUS 0

[[vk::binding(0)]] Texture2D<float4> hit0_tex;
[[vk::binding(1)]] Texture2D<float4> half_view_normal_tex;
[[vk::binding(2)]] Texture2D<float> half_depth_tex;
[[vk::binding(3)]] Texture2D<float4> ssao_tex;
[[vk::binding(4)]] RWTexture2D<float4> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 output_tex_size;
    int4 spatial_resolve_offsets[16 * 4 * 8];
};

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    #if 1
        output_tex[px] = hit0_tex[px];
        return;
    #endif
    
    float4 sum = 0;
    float ex = 0;
    float ex2 = 0;

    const float3 center_normal_vs = half_view_normal_tex[px].rgb;
    const float center_depth = half_depth_tex[px];
    const float4 center_val_valid_packed = hit0_tex[px];
    const float3 center_val = center_val_valid_packed.rgb;
    const float center_validity = center_val_valid_packed.a;
    const float center_ssao = ssao_tex[px * 2].r;
    const float rel_std_dev = 1;//abs(center_validity);

    const float2 uv = get_uv(px, output_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, center_depth);

    #define USE_POISSON 1

    float spatial_sharpness = 0.25;//lerp(0.5, 0.25, saturate(center_ssao));
    #if !USE_POISSON

    int k = 2;
    int skip = 3;

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            const int2 sample_offset = int2(x, y) * skip;
    #else

    const int sample_count = 16;
    //const int sample_count = 1;
    const uint px_idx_in_quad = (((px.x & 1) | (px.y & 1) * 2) + frame_constants.frame_index) & 3;

    const bool input_gi_stable = center_validity >= 0.5;

    #if USE_DYNAMIC_KERNEL_RADIUS
        const uint filter_idx =
            input_gi_stable
                ? uint(clamp(center_ssao * 3.0, 0.0, 7.0))
                : 7;
    #else
        const uint filter_idx = 5;
    #endif

    {
        for (uint sample_i = 0; sample_i < sample_count; ++sample_i) {
            const int2 sample_offset = spatial_resolve_offsets[(px_idx_in_quad * 16 + sample_i) + 64 * filter_idx].xy;
    #endif

            const int2 sample_px = px + sample_offset;

            const float3 sample_normal_vs = half_view_normal_tex[sample_px].rgb;
            const float sample_depth = half_depth_tex[sample_px];
            const float3 sample_val = hit0_tex[sample_px].rgb;
            const float sample_ssao = ssao_tex[sample_px * 2].r;

            if (sample_depth != 0) {
                float wt = 1;
                wt *= exp2(-spatial_sharpness * sqrt(float(dot(sample_offset, sample_offset))));
                wt *= pow(saturate(dot(center_normal_vs, sample_normal_vs)), 20);
                wt *= exp2(-200.0 * abs(center_normal_vs.z * (center_depth / sample_depth - 1.0)));

                #if USE_SSAO_STEERING
                    wt *= exp2(-20.0 * abs(sample_ssao - center_ssao));
                #endif

                sum += float4(sample_val, 1) * wt;
                
                float luma = calculate_luma(sample_val);
                ex += luma * wt;
                ex2 += luma * luma * wt;
            }

            // Adaptive stopping
            if (sample_i >= 3) {
                float var = abs(ex2 / sum.a - (ex / sum.a) * (ex / sum.a));
                var *= (sample_i + 1) / sample_i;   // Bessel's correction, 0-based
                float rel_dev = sqrt(var) / (abs(ex) / sum.a);
                if (rel_dev < 0.3)
                {
                    break;
                }
            }
        }
    }

    float3 center_sample = hit0_tex[px].rgb;
    float norm_factor = 1.0 / max(1e-5, sum.a);

    float3 filtered = sum.rgb * norm_factor;
    ex *= norm_factor;
    ex2 *= norm_factor;
    float variance = max(0.0, ex2 - ex * ex);
    float rel_dev = sqrt(variance);// / max(1e-5, ex);

    //filtered = rel_dev;
    //filtered = center_sample;
    //filtered = filter_idx / 7.0;

    //output_tex[px] = float4(filtered, /*rel_dev*/rel_std_dev);
    output_tex[px] = float4(filtered, /*rel_dev*/center_validity);
}
