#include "../inc/frame_constants.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] RWTexture3D<float4> indirect_tex;

float4 sample_direct_from(int3 vx, uint dir_idx) {
    if (any(vx < 0 || vx >= CSGI2_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 offset = int3(CSGI2_VOLUME_DIMS * dir_idx, 0, 0);
    float4 val = direct_tex[offset + vx];
    val.rgb /= max(0.01, val.a);
    return val;
}

float4 sample_indirect_from(int3 vx, uint dir_idx, uint subray) {
    if (any(vx < 0 || vx >= CSGI2_VOLUME_DIMS)) {
        return 0.0.xxxx;
    }

    const int3 offset = int3(CSGI2_VOLUME_DIMS * dir_idx, 0, 0);
    const int3 subray_offset = int3(0, subray * CSGI2_VOLUME_DIMS, 0);

    return float4(indirect_tex[offset + subray_offset + vx].rgb, 1);
}

// TODO: 3D textures on NV seem to be only tiled in the XY plane,
// meaning that the -Z and +Z sweeps are fast, but the others are slow.
// Might want to reshuffle the textures, and then unshuffle them in the "subray combine" pass.
[numthreads(8, 8, 1)]
void main(uint3 dispatch_vx : SV_DispatchThreadID, uint idx_within_group: SV_GroupIndex) {
    const uint direct_dir_idx = dispatch_vx.z;
    const uint indirect_dir_idx = direct_dir_idx;
    const int3 slice_dir = CSGI2_SLICE_DIRS[direct_dir_idx];

    // HACK: avoid the horizon due to high brightnes at sunset. TODO: convolve sky
    const float3 atmosphere_color = atmosphere_default(normalize(CSGI2_INDIRECT_DIRS[indirect_dir_idx].xyz + float3(0.0, 0.05, 0.0)), SUN_DIRECTION);

    const int3 indirect_offset = int3(CSGI2_VOLUME_DIMS * indirect_dir_idx, 0, 0);

    static const uint TANGENT_COUNT = 4;
    uint tangent_dir_indices[TANGENT_COUNT];
    {for (uint i = 0; i < TANGENT_COUNT; ++i) {
        tangent_dir_indices[i] = ((direct_dir_idx & uint(~1)) + 2 + i) % CSGI2_SLICE_COUNT;
    }}

    int slice_z_start = (direct_dir_idx & 1) ? (CSGI2_VOLUME_DIMS - 1) : 0; 

    int3 initial_vx;
    if (direct_dir_idx < 2) {
        initial_vx = int3(slice_z_start, dispatch_vx.x, dispatch_vx.y);
    } else if (direct_dir_idx < 4) {
        initial_vx = int3(dispatch_vx.x, slice_z_start, dispatch_vx.y);
    } else {
        initial_vx = int3(dispatch_vx.x, dispatch_vx.y, slice_z_start);
    }

    uint rng = hash1(frame_constants.frame_index);
    const float jitter_amount = 0.0;
    const float2 subray_jitter = float2((uint_to_u01_float(hash1_mut(rng)) - 0.5), (uint_to_u01_float(hash1_mut(rng)) - 0.5)) * jitter_amount;

    static const float skew = 0.5;
    static const float2 subray_bias[4] = {
        float2(-skew, -skew) + subray_jitter,
        float2(skew, -skew) + subray_jitter,
        float2(-skew, skew) + subray_jitter,
        float2(skew, skew) + subray_jitter
    };

    //[unroll] for (uint subray = 0; subray < 4; ++subray) {
    { uint subray = frame_constants.frame_index % 4;
        float bias_x = subray_bias[subray].x;
        float bias_y = subray_bias[subray].y;
        float weights[4] = {
            max(0.0, 1.0 - bias_x),
            max(0.0, 1.0 + bias_x),
            max(0.0, 1.0 - bias_y),
            max(0.0, 1.0 + bias_y),
        };

        int3 vx = initial_vx;
        {[loop]
        for (uint slice_z = 0; slice_z < CSGI2_VOLUME_DIMS; ++slice_z, vx -= slice_dir) {
            float3 scatter = 0.0;
            float scatter_wt = 0.0;

            const float4 center_direct_s = sample_direct_from(vx, direct_dir_idx);

            if (center_direct_s.a == 1) {
                scatter = center_direct_s.rgb;
                scatter_wt = 1;
            } else {
                scatter = 0 == slice_z ? atmosphere_color : sample_indirect_from(vx + slice_dir, direct_dir_idx, subray).rgb;
                scatter_wt += 1;

                [unroll]
                for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
                    const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
                    const int3 tangent_dir = CSGI2_SLICE_DIRS[tangent_dir_idx];

                    const float center_opacity_t = sample_direct_from(vx, tangent_dir_idx).a;
                    const float4 direct_neighbor_t = sample_direct_from(vx + tangent_dir, tangent_dir_idx);
                    const float4 direct_neighbor_s = sample_direct_from(vx + tangent_dir, direct_dir_idx);

                    float3 neighbor_radiance = 0 == slice_z ? atmosphere_color : sample_indirect_from(vx + slice_dir + tangent_dir, indirect_dir_idx, subray).rgb;
                    
                    neighbor_radiance = lerp(neighbor_radiance, direct_neighbor_s.rgb, direct_neighbor_s.a);
                    // HACK: ad-hoc scale for off-axis contributions
                    neighbor_radiance = lerp(neighbor_radiance, 0.25*direct_neighbor_t.rgb, direct_neighbor_t.a);
                    neighbor_radiance = lerp(neighbor_radiance, 0.0.xxx, center_opacity_t);
                    neighbor_radiance = lerp(neighbor_radiance, 0.0.xxx, center_direct_s.a);

                    float wt = weights[tangent_i];
                    scatter += neighbor_radiance * wt;
                    scatter_wt += wt;
                }
            }

            float4 radiance = float4(scatter / max(scatter_wt, 1), 1);
            const int3 subray_offset = int3(0, subray * CSGI2_VOLUME_DIMS, 0);
            indirect_tex[subray_offset + vx + indirect_offset] = radiance;
        }}
    }
}
