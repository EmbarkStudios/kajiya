#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/sun.hlsl"

#include "../inc/atmosphere.hlsl"
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> integr_tex;
[[vk::binding(1)]] RWTexture3D<float4> cascade0_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 SLICE_DIRS[16];
}

#define USE_RAY_JITTER 1
#define USE_MULTIBOUNCE 1

static const float SKY_DIST = 1e5;

float3 vx_to_pos(float3 vx, float3x3 slice_rot) {
    return mul(slice_rot, vx - (GI_VOLUME_DIMS - 1.0) / 2.0) * GI_VOXEL_SIZE + gi_volume_center(slice_rot);
}

// HACK; broken
#define CSGI_LOOKUP_NEAREST_ONLY
#define alt_cascade0_tex cascade0_tex
#include "lookup.hlsl"


[numthreads(8, 8, 1)]
void main(uint3 px : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint grid_idx = px.x / GI_VOLUME_DIMS;
    px.x %= GI_VOLUME_DIMS;

    static const uint DIR_COUNT = 9;
    static const int3 dirs[DIR_COUNT] = {
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

    float3 scatter = 0.0;
    for (uint dir_i = 0; dir_i < DIR_COUNT; ++dir_i) {
        //const int3 preintegr_px = px * int3(1, DIR_COUNT, 1) + int3(GI_VOLUME_DIMS * grid_idx, dir_i, 0);
        const int3 preintegr_px = px + int3(GI_VOLUME_DIMS * grid_idx, GI_VOLUME_DIMS * dir_i, 0);

        const float4 color_transp = integr_tex[preintegr_px];

        scatter += color_transp.rgb;

        int3 src_px = int3(px) + dirs[dir_i];
        if (src_px.x >= 0 && src_px.x < GI_VOLUME_DIMS) {
            scatter += cascade0_tex[src_px + int3(GI_VOLUME_DIMS * grid_idx, 0, 0)].rgb * color_transp.a;
        }
    }
    float3 radiance = scatter.xyz / DIR_COUNT;

    float4 prev = cascade0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx];
    float4 cur = float4(radiance, 1);
    //float4 output = lerp(prev, cur, 0.2);
    float4 output = cur;
    cascade0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx] = output;
}
