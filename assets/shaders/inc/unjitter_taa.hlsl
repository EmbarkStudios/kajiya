#ifndef UNJITTER_TAA_HLSL
#define UNJITTER_TAA_HLSL

#include "image.hlsl"

struct UnjitteredSampleInfo {
    float3 color;
    float coverage;
    float3 ex;
    float3 ex2;
};

/// trait Remap {
///     float4 remap(float4 v);
/// }
template<typename Remap>
UnjitteredSampleInfo sample_image_unjitter_taa(
    TextureImage img,
    int2 output_px,
    float2 output_tex_size,
    float2 sample_offset_pixels,
    Remap remap = IdentityImageRemap::create()
) {
    const float2 input_tex_size = float2(img.size());
    const float2 input_resolution_scale = input_tex_size / output_tex_size;
    const int2 base_src_px = int2((output_px + 0.5) * input_resolution_scale);

    // In pixel units of the destination (upsampled)
    const float2 dst_sample_loc = float2(output_px) + 0.5;
    const float2 base_src_sample_loc = (base_src_px + 0.5) / input_resolution_scale;

    float4 res = 0.0;
    float3 ex = 0.0;
    float3 ex2 = 0.0;
    float dev_wt_sum = 0.0;

    // Stretch the kernel if samples become too sparse due to drastic upsampling
    const float kernel_distance_mult = min(1.0, 1.2 * input_tex_size.x / output_tex_size.x);

    int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            int2 src_px = base_src_px + int2(x, y);
            float2 src_sample_loc = base_src_sample_loc + float2(x, y) / input_resolution_scale;

            float4 col = remap.remap(img.fetch(src_px));
            float2 sample_center_offset = -sample_offset_pixels * float2(1, -1) / input_resolution_scale - (src_sample_loc - dst_sample_loc);

            float dist2 = dot(sample_center_offset, sample_center_offset);
            float dist = sqrt(dist2);

            float wt = mitchell_netravali(dist * kernel_distance_mult);
            float dev_wt = exp2(-dist2);

            res += col * wt;

            ex += col.xyz * dev_wt;
            ex2 += col.xyz * col.xyz * dev_wt;
            dev_wt_sum += dev_wt;
        }
    }

    const int sample_count = (k * 2 + 1) * (k * 2 + 1);
    const float ideal_coverage = 1.0 + 4 * 0.5 + 4 * 0.25;

    float2 sample_center_offset = -sample_offset_pixels / input_resolution_scale * float2(1, -1) - (base_src_sample_loc - dst_sample_loc);

    UnjitteredSampleInfo info;
    info.color = res.rgb / max(1e-5, res.a);
    //info.color = res.rgb / ideal_coverage * 1.5;
    //info.coverage = res.a / sample_count;
    //info.coverage = dev_wt_sum;
    info.coverage = res.a / sample_count;//exp2(-2 * dot(sample_center_offset, sample_center_offset) * kernel_distance_mult);
    info.ex = ex / dev_wt_sum;
    info.ex2 = ex2 / dev_wt_sum;
    return info;
}

#endif // UNJITTER_TAA_HLSL
