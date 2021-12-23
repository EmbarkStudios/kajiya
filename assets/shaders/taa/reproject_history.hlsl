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
[[vk::binding(4)]] RWTexture2D<float2> closest_velocity_output;
[[vk::binding(5)]] cbuffer _ {
    float4 input_tex_size;
    float4 output_tex_size;
};

// Optimization: Try to skip velocity dilation if velocity diff is small
// around the pixel.
#define APPROX_SKIP_DILATION true

float4 fetch_history(float2 uv) {
    float4 h = history_tex.SampleLevel(sampler_lnc, uv, 0);
	return float4(decode_rgb(h.xyz), h.w);
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
void main(
    uint2 px: SV_DispatchThreadID,
    uint idx_within_group: SV_GroupIndex,
    uint2 group_id: SV_GroupID
) {
    const float2 input_resolution_scale = input_tex_size.xy / output_tex_size.xy;
    const uint2 reproj_px = uint2((px + 0.5) * input_resolution_scale);

    float2 uv = get_uv(px, output_tex_size);
    uint2 closest_px = reproj_px;

#if APPROX_SKIP_DILATION
    // Find the bounding box of velocities around this 3x3 region
    float2 vel_min;
    float2 vel_max;
    {
        float2 v = reprojection_tex[reproj_px + int2(-1, -1)].xy;
        vel_min = v;
        vel_max = v;
    }
    {
        float2 v = reprojection_tex[reproj_px + int2(1, -1)].xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }
    {
        float2 v = reprojection_tex[reproj_px + int2(-1, 1)].xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }
    {
        float2 v = reprojection_tex[reproj_px + int2(1, 1)].xy;
        vel_min = min(vel_min, v);
        vel_max = max(vel_max, v);
    }

    bool should_dilate = any((vel_max - vel_min) > 0.1 * max(input_tex_size.zw, abs(vel_max + vel_min)));
    
    // Since we're only checking a few pixels, there's a chance we'll miss something.
    // Dilate in the wave to reduce the chance of that happening.
    //should_dilate |= WaveReadLaneAt(should_dilate, WaveGetLaneIndex() ^ 1);
    should_dilate |= WaveReadLaneAt(should_dilate, WaveGetLaneIndex() ^ 2);
    //should_dilate |= WaveReadLaneAt(should_dilate, WaveGetLaneIndex() ^ 8);
    should_dilate |= WaveReadLaneAt(should_dilate, WaveGetLaneIndex() ^ 16);

    // We want to find the velocity of the pixel which is closest to the camera,
    // which is critical to anti-aliased moving edges.
    // At the same time, when everything moves with roughly the same velocity
    // in the neighborhood of the pixel, we'd be performing this depth-based kernel
    // only to return the same value.
    // Therefore, we predicate the search on there being any appreciable
    // velocity difference around the target pixel. This ends up being faster on average.
    if (should_dilate)
#endif
    {
        float reproj_depth = depth_tex[reproj_px];
        int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float d = depth_tex[reproj_px + int2(x, y)];
                if (d > reproj_depth) {
                    reproj_depth = d;
                    closest_px = reproj_px + int2(x, y);
                }
            }
        }
    }

    const float2 reproj_xy = reprojection_tex[closest_px].xy;
    closest_velocity_output[px] = reproj_xy;
    float2 history_uv = uv + reproj_xy;

#if 0
    float4 history_packed = image_sample_catmull_rom(
        TextureImage::from_parts(history_tex, output_tex_size.xy),
        history_uv,
        HistoryRemap::create()
    );
#elif 1
    float4 history_packed = image_sample_catmull_rom_5tap(
        history_tex, sampler_llc, history_uv, output_tex_size.xy, HistoryRemap::create()
    );
#else
    float4 history_packed = fetch_history(history_uv);
#endif

    float3 history = history_packed.rgb;
    float history_coverage = max(0.0, history_packed.a);

    output_tex[px] = float4(history, history_coverage);
}
