#include "../inc/frame_constants.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/math.hlsl"

#include "common.hlsl"

[[vk::binding(0)]] RWTexture3D<float4> out0_tex;
[[vk::binding(1)]] Texture3D<float4> pretraced_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 SLICE_DIRS[16];
    float4 PRETRACE_DIRS[32];
}

float3 vx_to_pos(float3 vx, float3x3 slice_rot) {
    return mul(slice_rot, vx - (GI_VOLUME_DIMS - 1.0) / 2.0) * GI_VOLUME_SCALE + GI_VOLUME_CENTER;
}

float3 pos_to_pretrace_vx(float3 pos, float3x3 slice_rot) {
    pos -= GI_VOLUME_CENTER;
    pos /= GI_VOLUME_SCALE;
    pos = mul(pos, slice_rot);
    pos += (GI_PRETRACE_DIMS - 1.0) / 2.0;
    return pos;
}

[numthreads(8, 8, 1)]
void main(in uint3 px : SV_DispatchThreadID) {
    //px.z = GI_VOLUME_DIMS - 1 - px.z;
    const uint grid_idx = px.x / GI_VOLUME_DIMS;
    px.x %= GI_VOLUME_DIMS;

    const float3x3 slice_rot = build_orthonormal_basis(SLICE_DIRS[grid_idx].xyz);
    const float3 slice_dir = mul(slice_rot, float3(0, 0, -1));    
    const float3 trace_origin = vx_to_pos(px + float3(0, 0, 0.5), slice_rot);

    const float spread0 = 1.0;
    const float spread1 = 0.7;

    float3 scatter = 0.0;

    static const uint DIR_COUNT = 9;
    static const int3 dirs[9] = {
        int3(0, 0, -1),

        int3(1, 0, -1),
        int3(-1, 0, -1),
        int3(0, 1, -1),
        int3(0, -1, -1),

        int3(1, 1, -1),
        int3(-1, 1, -1),
        int3(1, -1, -1),
        int3(-1, -1, -1)
    };
    static const float weights[9] = {
        1,
        spread0, spread0, spread0, spread0,
        spread1, spread1, spread1, spread1
    };

    float total_wt = 0;

    [unroll]
    for (uint dir_i = 0; dir_i < DIR_COUNT; ++dir_i)
    {
        //const uint dir_i = 0;
        total_wt += weights[dir_i];

        float3 r_dir = mul(slice_rot, float3(dirs[dir_i]));

        float highest_dot = -1;
        uint pretraced_idx = 0;

        for (uint i = 0; i < GI_PRETRACE_COUNT; ++i) {
            const float d = dot(r_dir, -PRETRACE_DIRS[i].xyz);
            if (d > highest_dot) {
                highest_dot = d;
                pretraced_idx = i;
            }
        }

        const float3x3 pretrace_rot = build_orthonormal_basis(PRETRACE_DIRS[pretraced_idx].xyz);
        const float3 pretrace_vx_f = pos_to_pretrace_vx(trace_origin, pretrace_rot);
        const int3 pretrace_vx = int3(trunc(pretrace_vx_f.xy), trunc(pretrace_vx_f.z - 0.5));

        float4 pretrace_packed = 0.0.xxxx;
        if (all(pretrace_vx >= 0) && all(pretrace_vx < GI_PRETRACE_DIMS)) {
            int3 pretrace_vx_resolved = pretrace_vx + int3(GI_PRETRACE_DIMS * pretraced_idx, 0, 0);
            pretrace_packed = pretraced_tex[pretrace_vx_resolved];

            // HACK
            if (pretrace_packed.w < 0.8) {
                pretrace_packed = pretraced_tex[pretrace_vx_resolved + int3(0, 0, 1)];
            }
        }

        bool is_hit = pretrace_packed.w > 0.0;
        if (is_hit) {
            float3 total_radiance = pretrace_packed.rgb;

            scatter += total_radiance * weights[dir_i];
        } else {
            int3 src_px = int3(px) + dirs[dir_i];
            if (src_px.x >= 0 && src_px.x < GI_VOLUME_DIMS) {
                scatter += out0_tex[src_px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx].rgb * weights[dir_i];
            }
        }
    }

    float3 radiance = scatter.xyz / total_wt;

    out0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx] = float4(radiance, 1);
}
