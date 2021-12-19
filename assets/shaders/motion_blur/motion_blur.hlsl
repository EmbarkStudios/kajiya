#include "../inc/samplers.hlsl"
#include "../inc/uv.hlsl"
#include "../inc/frame_constants.hlsl"

[[vk::binding(0)]] Texture2D<float3> input_tex;
[[vk::binding(1)]] Texture2D<float2> velocity_tex;
[[vk::binding(2)]] Texture2D<float2> tile_velocity_tex;
[[vk::binding(3)]] Texture2D<float> depth_tex;
[[vk::binding(4)]] RWTexture2D<float3> output_tex;
[[vk::binding(5)]] cbuffer _ {
    float4 depth_tex_size;
    float4 output_tex_size;
};

float2 depth_cmp(float center_depth, float sample_depth, float depth_scale) {
    return saturate(0.5 + float2(depth_scale, -depth_scale) * (sample_depth - center_depth));
}

float2 spread_cmp(float offset_len, float2 spread_len) {
    return saturate(spread_len - offset_len + 1.0);
}

float sample_weight(float center_depth, float sample_depth, float offset_len, float center_spread_len, float sample_spread_len, float depth_scale) {
    float2 dc = depth_cmp(center_depth, sample_depth, depth_scale);
    float2 sc = spread_cmp(offset_len, float2(center_spread_len, sample_spread_len));
    return dot(dc, sc);
}

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
#if 0
    output_tex[px] = input_tex[px];
    return;
#endif

    const float2 uv = get_uv(px, output_tex_size);
    float blur_scale = 0.5;

    // TODO: Fixed shutter time (add time_delta_seconds)
    blur_scale /= frame_constants.delta_time_seconds * 60.0;

    // Scramble tile coordinates to diffuse the tile quantization in noise
    int noise1;
    int2 tile_offset = px;
    {
        tile_offset.x += (tile_offset.x << 4);
        tile_offset.x ^= (tile_offset.x >> 6);
        tile_offset.y += (tile_offset.x << 1);
        tile_offset.y += (tile_offset.y << 6);
        tile_offset.y ^= (tile_offset.y >> 2);
        tile_offset.x ^= tile_offset.y;
        noise1 = tile_offset.x ^ (tile_offset.y << 1);
        tile_offset &= 31;
        tile_offset -= 15;
        noise1 &= 31;
        noise1 -= 15;
    }

    const int2 velocity_tile_coord = int2(uv * depth_tex_size.xy) + tile_offset;
    const float2 tile_velocity = blur_scale * tile_velocity_tex[clamp(velocity_tile_coord, int2(0, 0), int2(depth_tex_size.xy) - 1) / 16];

    const int kernel_width = 4;
    float noise = 0; {
        float scale = 0.25;
        float2 positionMod = float2(uint2(px) & 1u);
        noise = (-scale + 2.0 * scale * positionMod.x) * (-1.0 + 2.0 * positionMod.y);
    }

    noise = 0.5 * float(noise1) / 15;

    float center_offset_len = noise / kernel_width * 0.5;
    float2 center_uv = uv + tile_velocity * center_offset_len;

    float3 center_color = input_tex[clamp(int2(center_uv * output_tex_size.xy), int2(0, 0), int2(output_tex_size.xy - 1))];
    float center_depth = -depth_to_view_z(depth_tex.SampleLevel(sampler_nnc, center_uv, 0));
    float2 center_velocity = blur_scale * velocity_tex.SampleLevel(sampler_lnc, center_uv, 0);
    float2 center_velocity_px = center_velocity * depth_tex_size.xy;

    float4 sum = 0.0.xxxx;

    float soft_z = 16.0;

    float sample_count = 1.0f;
    if (length(tile_velocity) > 0) {
        for (int i = 1; i <= kernel_width; ++i) {
            float offset_len0 = float( i + noise) / kernel_width * 0.5;
            float offset_len1 = float(-i + noise) / kernel_width * 0.5;
            float2 uv0 = uv + tile_velocity * offset_len0;
            float2 uv1 = uv + tile_velocity * offset_len1;
            int2 px0 = int2(uv0 * depth_tex_size.xy);
            int2 px1 = int2(uv1 * depth_tex_size.xy);

            float d0 = -depth_to_view_z(depth_tex[px0]);
            float d1 = -depth_to_view_z(depth_tex[px1]);
            float v0 = length(blur_scale * velocity_tex[px0] * depth_tex_size.xy);
            float v1 = length(blur_scale * velocity_tex[px1] * depth_tex_size.xy);

            float weight0 = sample_weight(center_depth, d0, length((uv0 - uv) * depth_tex_size.xy), length(center_velocity_px), v0, soft_z);
            float weight1 = sample_weight(center_depth, d1, length((uv1 - uv) * depth_tex_size.xy), length(center_velocity_px), v1, soft_z);

            bool2 mirror = bool2(d0 > d1, v1 > v0);
            weight0 = all(mirror) ? weight1 : weight0;
            weight1 = any(mirror) ? weight1 : weight0;

            float valid0 = float(all(uv0 == clamp(uv0, 0.0, 1.0)));
            float valid1 = float(all(uv1 == clamp(uv1, 0.0, 1.0)));

            weight0 *= valid0;
            weight1 *= valid1;
            sample_count += valid0 + valid1;

            sum += float4(input_tex.SampleLevel(sampler_lnc, uv0, 0).rgb, 1) * weight0;
            sum += float4(input_tex.SampleLevel(sampler_lnc, uv1, 0).rgb, 1) * weight1;
        }

        sum *= 1.0 / sample_count;
    }

    float3 result = sum.rgb + (1.0 - sum.w) * center_color;
 
    output_tex[px] = result;
}