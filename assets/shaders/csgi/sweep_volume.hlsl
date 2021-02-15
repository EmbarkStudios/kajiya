#include "../inc/frame_constants.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/math.hlsl"

#include "common.hlsl"

[[vk::binding(0)]] RWTexture3D<float4> out0_tex;
[[vk::binding(1)]] Texture3D<float> pretrace_hit_tex;
[[vk::binding(2)]] Texture3D<float4> pretrace_col_tex;
[[vk::binding(3)]] Texture3D<float4> pretrace_normal_tex;
[[vk::binding(4)]] cbuffer _ {
    float4 SLICE_DIRS[GI_SLICE_COUNT];
    float4 PRETRACE_DIRS[GI_PRETRACE_COUNT];
    uint4 RAY_DIR_PRETRACE_INDICES[GI_SLICE_COUNT * 9];
}

float3 vx_to_pos(float3 vx, float3x3 slice_rot) {
    return mul(slice_rot, vx - (GI_VOLUME_DIMS - 1.0) / 2.0) * GI_VOXEL_SIZE + gi_volume_center(slice_rot);
}

float3 pos_to_pretrace_vx(float3 pos, float3x3 slice_rot) {
    pos -= gi_volume_center(slice_rot);
    pos /= ((GI_VOLUME_SIZE * 1.0) / GI_PRETRACE_DIMS);
    pos = mul(pos, slice_rot);
    pos += (GI_PRETRACE_DIMS - 1.0) / 2.0;
    return pos;
}

static const uint DIR_COUNT = 9;
groupshared float3 r_dirs[DIR_COUNT];
groupshared uint r_pretrace_indices[DIR_COUNT];

[numthreads(8, 8, 1)]
void main(uint3 px2d_grid : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint grid_idx = px2d_grid.z;

    const float3x3 slice_rot = build_orthonormal_basis(SLICE_DIRS[grid_idx].xyz);
    const float3 slice_dir = mul(slice_rot, float3(0, 0, -1));    

    const float spread0 = 1.0;
    const float spread1 = 1.0;

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

    if (idx_within_group < 9) {
        const uint dir_i = idx_within_group;
        float3 r_dir = normalize(mul(slice_rot, float3(dirs[dir_i])));
        r_dirs[dir_i] = r_dir;
        r_pretrace_indices[dir_i] = RAY_DIR_PRETRACE_INDICES[grid_idx * 9 + dir_i].x;
    }

    GroupMemoryBarrierWithGroupSync();

    [loop]
    for (uint px_z = 0; px_z < GI_VOLUME_DIMS; ++px_z) {
        const uint3 px = uint3(px2d_grid.xy, px_z);

        float3 scatter = 0.0;

        //px.z = GI_VOLUME_DIMS - 1 - px.z;

        //const float3 trace_origin = vx_to_pos(px + float3(0, 0, 0.5), slice_rot);
        const float3 trace_origin = vx_to_pos(px, slice_rot);

        float total_wt = 0;

        for (uint dir_i = 0; dir_i < DIR_COUNT; ++dir_i)
        {
            //const uint dir_i = 0;
            total_wt += weights[dir_i];

            const float3 r_dir = r_dirs[dir_i];

    #if 0
            uint pretraced_idx = 0;
            float highest_dot = -1;

            for (uint i = 0; i < GI_PRETRACE_COUNT; ++i) {
                const float d = dot(r_dir, -PRETRACE_DIRS[i].xyz);
                if (d > highest_dot) {
                    highest_dot = d;
                    pretraced_idx = i;
                }
            }
    #else
            const uint pretraced_idx = r_pretrace_indices[dir_i].x;
    #endif

            const float3 pretrace_dir = PRETRACE_DIRS[pretraced_idx].xyz;

            const float3x3 pretrace_rot = build_orthonormal_basis(pretrace_dir);
            const float3 pretrace_vx_f = pos_to_pretrace_vx(trace_origin, pretrace_rot);
            //const int3 pretrace_vx = int3(trunc(pretrace_vx_f.xy), trunc(pretrace_vx_f.z - 0.5));
            const int3 pretrace_vx = int3(trunc(pretrace_vx_f));
            const int3 pretrace_vx_resolved = pretrace_vx + int3(GI_PRETRACE_DIMS * pretraced_idx, 0, 0);

            float4 pretrace_packed = 0.0.xxxx;
            float3 hit_normal = 0.0.xxx;

            if (all(pretrace_vx >= 0) && all(pretrace_vx < GI_PRETRACE_DIMS)) {
                if (pretrace_hit_tex[pretrace_vx_resolved] > 0) {
                    pretrace_packed = pretrace_col_tex[pretrace_vx_resolved];
                    hit_normal = pretrace_normal_tex[pretrace_vx_resolved].xyz * 2 - 1;

                    // HACK
                    if (pretrace_packed.w > 1.5 * float(GI_PRETRACE_DIMS) / GI_VOLUME_DIMS) {
                        pretrace_packed = 0.0.xxxx;
                    }
                }
            }

            bool is_hit = pretrace_packed.w > 0.0;
            if (is_hit) {
                float3 total_radiance = pretrace_packed.rgb * smoothstep(0.0, 0.1, dot(-hit_normal, slice_dir));

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
}
