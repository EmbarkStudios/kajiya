#include "inc/frame_constants.hlsl"
#include "inc/uv.hlsl"

[[vk::binding(0)]] Texture2D<float> depth_tex;
[[vk::binding(1)]] Texture2D<float> prev_depth_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

    float3 eye_pos = mul(frame_constants.view_constants.view_to_world, float4(0, 0, 0, 1)).xyz;

    float depth = 0.0;
    {
        const int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float s_depth = depth_tex[px + int2(x, y)];
                if (s_depth != 0.0) {
                    depth = max(depth, s_depth);
                }
            }
        }
    }

    float4 pos_cs = float4(uv_to_cs(uv), depth, 1.0);
    float4 pos_vs = mul(frame_constants.view_constants.clip_to_view, pos_cs);

    float4 prev_cs_cur_depth = mul(frame_constants.view_constants.clip_to_prev_clip, pos_cs);
    float4 pos_prev_vs = mul(frame_constants.view_constants.prev_clip_to_prev_view, prev_cs_cur_depth);
    pos_prev_vs /= pos_prev_vs.w;

    prev_cs_cur_depth /= prev_cs_cur_depth.w;

    const float2 prev_uv = cs_to_uv(prev_cs_cur_depth.xy);
    const float2 uv_diff = prev_uv - uv;

    float validity = all(prev_cs_cur_depth.xy == clamp(
        prev_cs_cur_depth.xy,
        -1.0 + output_tex_size.zw,
        1.0 - output_tex_size.zw
    )) ? 1.0 : 0.0;

    float max_depth_validity = 0.0;
    {
        int2 prev_px = int2(prev_uv * output_tex_size.xy);
        const int k = 1;
        for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float prev_depth = prev_depth_tex[prev_px + int2(x, y)];
                float4 prev_cs = float4(prev_cs_cur_depth.xy, prev_depth, 1.0);
                float4 prev_vs = mul(frame_constants.view_constants.prev_clip_to_prev_view, prev_cs);
                prev_vs /= prev_vs.w;
                max_depth_validity = max(
                    max_depth_validity,
                    saturate(exp2(-4.0 * length(pos_prev_vs.xyz - prev_vs.xyz) / max(1e-3, min(-pos_prev_vs.z, -prev_vs.z))))
                );
            }
        }
    }
    validity *= max_depth_validity;

    float2 texel_center_offset = abs(0.5 - frac(prev_uv * output_tex_size.xy));
    float accuracy = 1.0 - texel_center_offset.x - texel_center_offset.y;

    output_tex[px] = float4(
        uv_diff,
        validity,
        accuracy
    );
}
