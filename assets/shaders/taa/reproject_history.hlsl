#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/color.hlsl"
#include "../inc/image.hlsl"
#include "../inc/frame_constants.hlsl"
#include "taa_common.hlsl"

[[vk::binding(0)]] Texture2D<float4> history_tex;
[[vk::binding(1)]] Texture2D<float4> reprojection_tex;
[[vk::binding(2)]] Texture2D<float> depth_tex;
[[vk::binding(3)]] RWTexture2D<float4> output_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

float3 fetch_history(float2 uv) {
	return decode_rgb(
        history_tex.SampleLevel(sampler_lnc, uv, 0).xyz
    );
}

struct HistoryRemap {
    static HistoryRemap create() {
        HistoryRemap res;
        return res;
    }

    float4 remap(float4 v) {
        return float4(decode_rgb(v.rgb), v.a);
    }
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    const float2 input_resolution_scale = input_tex_size.xy / output_tex_size.xy;
    float2 uv = get_uv(px, output_tex_size);

    const uint2 reproj_px = uint2((px + 0.5) * input_resolution_scale);
    float2 reproj_xy = reprojection_tex[reproj_px].xy;
    {
        float reproj_depth = depth_tex[reproj_px];
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float d = depth_tex[reproj_px + int2(x, y)];
                if (d > reproj_depth) {
                    reproj_depth = d;
                    reproj_xy = reprojection_tex[reproj_px + int2(x, y)].xy;
                }
            }
        }
    }
    
    //const float4 reproj = reprojection_tex[reproj_px];
    float2 history_uv = uv + reproj_xy;

#if 0
    float history_g = image_sample_catmull_rom(
        TextureImage::from_parts(history_tex, output_tex_size.xy),
        history_uv,
        HistoryRemap::create()
    ).y;
    float3 history = fetch_history(history_uv);
    if (history.y > 1e-5) {
        history *= history_g / history.y;
    }
#elif 1
    float4 history_packed = image_sample_catmull_rom(
        TextureImage::from_parts(history_tex, output_tex_size.xy),
        history_uv,
        HistoryRemap::create()
    );
    float3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);
    //float3 history = history_tex[px].rgb;
#else
    float4 history_packed = history_tex.SampleLevel(sampler_lnc, history_uv, 0);
    float3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);
#endif

    output_tex[px] = float4(history, history_coverage);
}
