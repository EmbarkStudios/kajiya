#include "inc/frame_constants.hlsl"
#include "inc/uv.hlsl"

[[vk::binding(0)]] Texture2D<float> depth_tex;
[[vk::binding(1)]] RWTexture2D<float4> output_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 output_tex_size;
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

    float3 eye_pos = mul(frame_constants.view_constants.view_to_world, float4(0, 0, 0, 1)).xyz;

    float depth = 0.0;
    const int k = 1;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float s_depth = depth_tex[px + int2(x, y)];
            if (s_depth != 0.0) {
                depth = max(depth, s_depth);
            }
        }
    }

    float4 pos_cs = float4(uv_to_cs(uv), depth, 1.0);

    //float4 pos_vs = mul(frame_constants.view_constants.clip_to_view, pos_cs);
    //float4 pos_ws = mul(frame_constants.view_constants.view_to_world, pos_vs);

    float4 prev_cs = mul(frame_constants.view_constants.clip_to_prev_clip, pos_cs);
    prev_cs /= prev_cs.w;

    //float4 prev_vs = mul(frame_constants.view_constants.prev_clip_to_prev_view, prev_cs);
    //float4 prev_ws = mul(frame_constants.view_constants.prev_view_to_prev_world, prev_vs);

    //pos_ws /= pos_ws.w;
    //prev_ws /= prev_ws.w;

    const float2 prev_uv = cs_to_uv(prev_cs.xy);
    const float2 uv_diff = prev_uv - uv;

    float validity = all(prev_cs.xy == clamp(
        prev_cs.xy,
        -1.0 + output_tex_size.zw,
        1.0 - output_tex_size.zw
    )) ? 1.0 : 0.0;

    float2 texel_center_offset = abs(0.5 - frac(prev_uv * output_tex_size.xy));
    float accuracy = 1.0 - texel_center_offset.x - texel_center_offset.y;

    output_tex[px] = float4(
        uv_diff,
        validity,
        accuracy
    );
}
