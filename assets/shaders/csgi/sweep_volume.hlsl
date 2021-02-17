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
[[vk::binding(1)]] RWTexture3D<float4> csgi_cascade0_tex;
[[vk::binding(2)]] cbuffer _ {
    float4 CSGI_SLICE_DIRS[16];
}

[numthreads(8, 8, 1)]
void main(uint2 px_2d : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint grid_idx = px_2d.x / GI_VOLUME_DIMS;
    px_2d.x %= GI_VOLUME_DIMS;

    [loop]
    for (uint slice_z = 0; slice_z < GI_VOLUME_DIMS; ++slice_z) {
        uint3 px = uint3(px_2d, slice_z);
        float3 scatter = 0.0;

        [unroll]
        for (uint dir_i = 0; dir_i < GI_NEIGHBOR_DIR_COUNT; ++dir_i) {
            const int3 preintegr_px = px + int3(GI_VOLUME_DIMS * grid_idx, GI_VOLUME_DIMS * dir_i, 0);
            const float4 color_transp = integr_tex[preintegr_px];

            scatter += color_transp.rgb;

            if (0 == slice_z) {
                // Sample lighting coming in from the outside of the cascade.
                // TODO: sample other cascades, do a longer trace to find if sky is blocked, etc.
                scatter += atmosphere_default(-CSGI_SLICE_DIRS[dir_i].xyz, SUN_DIRECTION) * color_transp.a;
            } else {
                int3 src_px = int3(px) + GI_NEIGHBOR_DIRS[dir_i];
                if (src_px.x >= 0 && src_px.x < GI_VOLUME_DIMS) {
                    float3 neighbor_radiance = csgi_cascade0_tex[src_px + int3(GI_VOLUME_DIMS * grid_idx, 0, 0)].rgb;
                    scatter += neighbor_radiance * color_transp.a;
                }
            }
        }

        csgi_cascade0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx] = float4(scatter.xyz / GI_NEIGHBOR_DIR_COUNT, 1);
    }
}
