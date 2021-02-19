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
[[vk::binding(2)]] RWTexture3D<float4> csgi_cascade0_suppressed_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 CSGI_SLICE_DIRS[16];
}

[numthreads(8, 8, 1)]
void main(uint2 px_2d : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint grid_idx = px_2d.x / GI_VOLUME_DIMS;
    px_2d.x %= GI_VOLUME_DIMS;

    float3 atmosphere_color = atmosphere_default(-CSGI_SLICE_DIRS[grid_idx].xyz, SUN_DIRECTION);

    {[loop]
    for (uint slice_z = 0; slice_z < GI_VOLUME_DIMS; ++slice_z) {
        uint3 px = uint3(px_2d, slice_z);
        float3 scatter = 0.0;
        float scatter_wt = 0.0;
        float unoccluded = 1.0;

        [unroll]
        for (uint dir_i = 0; dir_i < GI_NEIGHBOR_DIR_COUNT; ++dir_i) {
            const int3 preintegr_px = px + int3(GI_VOLUME_DIMS * grid_idx, GI_VOLUME_DIMS * dir_i, 0);
            float4 color_transp = integr_tex[preintegr_px];

            if (color_transp.x < 0) {
                continue;
            }

            scatter += color_transp.rgb;
            scatter_wt += 1;

            if (0 == slice_z) {
                // Sample lighting coming in from the outside of the cascade.
                // TODO: sample other cascades, do a longer trace to find if sky is blocked, etc.
                scatter += atmosphere_color * color_transp.a;
            } else {
                int3 src_px = int3(px) + GI_NEIGHBOR_DIRS[dir_i];                
                if (all(src_px >= 0 && src_px < GI_VOLUME_DIMS)){
                    float3 neighbor_radiance = csgi_cascade0_tex[src_px + int3(GI_VOLUME_DIMS * grid_idx, 0, 0)].rgb;
                    scatter += neighbor_radiance * color_transp.a;
                } else {
                    //scatter += atmosphere_color * color_transp.a;
                }

                if (0 == dir_i) {
                    unoccluded *= color_transp.a;
                }
            }
        }

        float4 radiance = float4(scatter.xyz / max(scatter_wt, 1), 1);

        csgi_cascade0_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx] = radiance;

        // Leaks can happen when sampling from cells that had hits.
        // This reduces the amount of leaking, at the cost of some darkening.
        csgi_cascade0_suppressed_tex[px + int3(GI_VOLUME_DIMS, 0, 0) * grid_idx] = radiance * unoccluded;
    }}
}
