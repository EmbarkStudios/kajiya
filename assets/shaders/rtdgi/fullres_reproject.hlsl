#include "../inc/samplers.hlsl"
#include "../inc/color.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/bilinear.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/image.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] Texture2D<float4> reprojection_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 output_tex_size;
};

static const bool USE_SHARPENING_HISTORY_FETCH = true;

// For `image_sample_catmull_rom`. Not applying the actual color remap here to reduce cost.
struct HistoryRemap {
    static HistoryRemap create() {
        HistoryRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return v;
    }
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    
    float4 center = input_tex[px];
    float4 reproj = reprojection_tex[px];
    float2 prev_uv = uv + reproj.xy;

    uint quad_reproj_valid_packed = uint(reproj.z * 15.0 + 0.5);

    // For the sharpening (4x4) kernel, we need to know whether our neighbors are valid too,
    // as otherwise we end up over-sharpening with fake history (moving edges rather than scene features).
    const uint4 reproj_valid_neigh =
        uint4(reprojection_tex.GatherBlue(sampler_nnc, uv + 0.5 * sign(prev_uv) * output_tex_size.zw) * 15.0 + 0.5);

    float4 history = 0.0.xxxx;

    if (0 == quad_reproj_valid_packed) {
        // Everything invalid
    } else if (15 == quad_reproj_valid_packed) {
        if (USE_SHARPENING_HISTORY_FETCH && all(reproj_valid_neigh == 15)) {
            history = max(0.0.xxxx, image_sample_catmull_rom(
                TextureImage::from_parts(input_tex, output_tex_size.xy),
                prev_uv,
                HistoryRemap::create()
            ));
        } else {
            history = input_tex.SampleLevel(sampler_lnc, prev_uv, 0);
        }
    } else {
        // Only some samples are valid. Only include those, and don't do a sharpening fetch here.

        float4 quad_reproj_valid = (quad_reproj_valid_packed & uint4(1, 2, 4, 8)) != 0;

        const Bilinear bilinear = get_bilinear_filter(prev_uv, output_tex_size.xy);
        float4 s00 = input_tex[int2(bilinear.origin) + int2(0, 0)];
        float4 s10 = input_tex[int2(bilinear.origin) + int2(1, 0)];
        float4 s01 = input_tex[int2(bilinear.origin) + int2(0, 1)];
        float4 s11 = input_tex[int2(bilinear.origin) + int2(1, 1)];

        float4 weights = get_bilinear_custom_weights(bilinear, quad_reproj_valid);

        if (dot(weights, 1.0) > 1e-5) {
            history = apply_bilinear_custom_weights(s00, s10, s01, s11, weights);
        }
    }

    output_tex[px] = history;
}
