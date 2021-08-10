#include "../inc/samplers.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/math.hlsl"
#include "common.hlsl"

[[vk::binding(0)]] Texture3D<float4> direct_tex;
[[vk::binding(1)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(2)]] Texture3D<float> direct_opacity_tex;
[[vk::binding(3)]] Texture3D<float3> outer_cascade_subray_indirect_tex;
[[vk::binding(4)]] RWTexture3D<float3> subray_indirect_tex;
[[vk::binding(5)]] RWTexture3D<float3> indirect_tex;
[[vk::binding(6)]] cbuffer _ {
    uint __cascade_idx;
    uint quantum_idx;
}

#define USE_DEEP_OCCLUDE 1

static const uint SUBRAY_COUNT = CSGI_DIAGONAL_SUBRAY_COUNT;

float4 sample_direct_from(uint cascade_idx, int3 vx, uint dir_idx) {
    if (!gi_volume_contains_vx(frame_constants.gi_cascades[cascade_idx], vx)) {
        return 0.0.xxxx;
    }
    vx = csgi_wrap_vx_within_cascade(vx);

    const int3 offset = int3(CSGI_VOLUME_DIMS * dir_idx, 0, 0);
    float4 val = direct_tex[offset + vx];
    val.rgb /= max(0.01, val.a);
    return max(0.0, val);
}

float3 sample_outer_cascade_subray_indirect_from(uint cascade_idx, int3 vx, uint dir_idx, uint subray, float3 fallback_color) {
    if (!gi_volume_contains_vx(frame_constants.gi_cascades[cascade_idx + 1], vx)) {
        return fallback_color;
    }

    return outer_cascade_subray_indirect_tex[
        csgi_diagonal_vx_dir_subray_to_subray_vx(csgi_wrap_vx_within_cascade(vx), dir_idx, subray)
    ];
}

float3 sample_subray_indirect_from(uint cascade_idx, int3 vx, uint dir_idx, uint subray, float3 fallback_color) {
    // Check if we're out of the cascade, and if so, in which direction.
    // We'll use this to shift the lookup into the parent cascade, favoring
    // sampling towards the inside of the smaller cascade. This avoids leaks
    // when inside buildings, making sure the lookups don't sample the outside.
    const int3 outlier_offset = gi_volume_get_cascade_outlier_offset(frame_constants.gi_cascades[cascade_idx], vx);

    if (any(outlier_offset != 0)) {
        if (cascade_idx < CSGI_CASCADE_COUNT - 1) {
            // Note: Only works when the cascade exponential scaling is 2.0; Search token: b518ed19-c715-4cc7-9bc7-e0dbbca3e037
            return sample_outer_cascade_subray_indirect_from(
                cascade_idx,
                // Control rounding, biasing towards the direction of light flow,
                // in order not to jump over small blockers.
                (vx + CSGI_INDIRECT_DIRS[dir_idx]) / 2 - outlier_offset,
                dir_idx,
                subray,
                fallback_color);
        } else {
            return fallback_color;
        }
    }
    return subray_indirect_tex[
        csgi_diagonal_vx_dir_subray_to_subray_vx(csgi_wrap_vx_within_cascade(vx), dir_idx, subray)
    ];
}

void write_subray_indirect_to(float3 radiance, int3 vx, uint dir_idx, uint subray) {
    subray_indirect_tex[
        csgi_diagonal_vx_dir_subray_to_subray_vx(vx, dir_idx, subray)
    ] = prequant_shift_11_11_10(radiance);
}

void diagonal_sweep_volume(const uint2 dispatch_thread_id, const uint indirect_dir_idx, const uint cascade_idx) {
    const int3 indirect_dir = CSGI_INDIRECT_DIRS[indirect_dir_idx];
    const uint dir_i_idx = 0 + (indirect_dir.x > 0 ? 1 : 0);
    const uint dir_j_idx = 2 + (indirect_dir.y > 0 ? 1 : 0);
    const uint dir_k_idx = 4 + (indirect_dir.z > 0 ? 1 : 0);
    const int3 dir_i = CSGI_DIRECT_DIRS[dir_i_idx];
    const int3 dir_j = CSGI_DIRECT_DIRS[dir_j_idx];
    const int3 dir_k = CSGI_DIRECT_DIRS[dir_k_idx];

    float3 atmosphere_color = sky_cube_tex.SampleLevel(sampler_llr, CSGI_INDIRECT_DIRS[indirect_dir_idx].xyz, 0).rgb;

    static const uint PLANE_COUNT = CSGI_VOLUME_DIMS * 3 - 2;

    #if 1
    static const uint plane_start_idx = (frame_constants.frame_index % 2) * PLANE_COUNT / 2;
    static const uint plane_end_idx = (frame_constants.frame_index % 2 + 1) * PLANE_COUNT / 2;
    #else
    static const uint plane_start_idx = 0;
    static const uint plane_end_idx = PLANE_COUNT;
    #endif

    {[loop]
    for (uint plane_idx = plane_start_idx; plane_idx < plane_end_idx; ++plane_idx) {
        // A diagonal cross-section of a 3d grid creates a 2d hexagonal grid.
        // Here, axial coordinates are used to find the cells which belong to each slice.
        // See: https://www.redblobgames.com/grids/hexagons/

        const int sum_to = plane_idx;
        const int extent = CSGI_VOLUME_DIMS;
        const int xmin = max(0, sum_to - (extent - 1) * 2);
        const int vx_x = dispatch_thread_id.x + xmin;
        const int ymin = max(0, sum_to - (extent - 1) - vx_x);
        const int vx_y = dispatch_thread_id.y + ymin;
        int3 dispatch_vx = int3(vx_x, vx_y, plane_idx - vx_x - vx_y);
        dispatch_vx = indirect_dir > 0 ? (CSGI_VOLUME_DIMS - dispatch_vx - 1) : dispatch_vx;

        const int3 vx = csgi_dispatch_vx_to_global_vx(dispatch_vx, cascade_idx);

        [branch]
        if (all(dispatch_vx >= 0 && dispatch_vx < extent)) {
            float4 center_direct_i = sample_direct_from(cascade_idx, vx, dir_i_idx);
            float4 center_direct_j = sample_direct_from(cascade_idx, vx, dir_j_idx);
            float4 center_direct_k = sample_direct_from(cascade_idx, vx, dir_k_idx);

            #if USE_DEEP_OCCLUDE
            {
                const float4 center_direct_i2 = sample_direct_from(cascade_idx, vx + dir_i, dir_i_idx);
                const float4 center_direct_j2 = sample_direct_from(cascade_idx, vx + dir_j, dir_j_idx);
                const float4 center_direct_k2 = sample_direct_from(cascade_idx, vx + dir_k, dir_k_idx);

                center_direct_i = prelerp(center_direct_i, center_direct_i2);
                center_direct_j = prelerp(center_direct_j, center_direct_j2);
                center_direct_k = prelerp(center_direct_k, center_direct_k2);
            }
            #endif

            static const float3 subray_wts[SUBRAY_COUNT] = CSGI_DIAGONAL_SUBRAY_TANGENT_WEIGHTS;
            static const float subray_wt_sum = dot(subray_wts[0], 1.0.xxx);

            float3 subray_radiance[SUBRAY_COUNT] = {
                0.0.xxx, 0.0.xxx, 0.0.xxx,
            };

// Equivalent; speed varies depending on optimizer mood
#if 0
            {for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = sample_subray_indirect_from(cascade_idx, vx + dir_i, indirect_dir_idx, subray, atmosphere_color);

                float wt = subray_wts[subray].x;
                indirect = lerp(indirect, center_direct_i.rgb, center_direct_i.a);
                subray_radiance[subray] += indirect * wt / subray_wt_sum;
            }}

            {for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = sample_subray_indirect_from(cascade_idx, vx + dir_j, indirect_dir_idx, subray, atmosphere_color);

                float wt = subray_wts[subray].y;
                indirect = lerp(indirect, center_direct_j.rgb, center_direct_j.a);
                subray_radiance[subray] += indirect * wt / subray_wt_sum;
            }}

            {for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = sample_subray_indirect_from(cascade_idx, vx + dir_k, indirect_dir_idx, subray, atmosphere_color);

                float wt = subray_wts[subray].z;
                indirect = lerp(indirect, center_direct_k.rgb, center_direct_k.a);
                subray_radiance[subray] += indirect * wt / subray_wt_sum;
            }}
#else
            {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect_i = sample_subray_indirect_from(cascade_idx, vx + dir_i, indirect_dir_idx, subray, atmosphere_color);
                float3 indirect_j = sample_subray_indirect_from(cascade_idx, vx + dir_j, indirect_dir_idx, subray, atmosphere_color);
                float3 indirect_k = sample_subray_indirect_from(cascade_idx, vx + dir_k, indirect_dir_idx, subray, atmosphere_color);

                indirect_i = lerp(indirect_i, center_direct_i.rgb, center_direct_i.a);
                indirect_j = lerp(indirect_j, center_direct_j.rgb, center_direct_j.a);
                indirect_k = lerp(indirect_k, center_direct_k.rgb, center_direct_k.a);

                const float3 wt = subray_wts[subray];
                subray_radiance[subray] = (indirect_i * wt.x + indirect_j * wt.y + indirect_k * wt.z) / subray_wt_sum;
            }}
#endif

            float3 combined_indirect = 0.0.xxx;

            {[unroll] for (uint subray = 0; subray < SUBRAY_COUNT; ++subray) {
                float3 indirect = subray_radiance[subray];
                write_subray_indirect_to(indirect, dispatch_vx, indirect_dir_idx, subray);
                combined_indirect += indirect;
            }}

            #if CSGI_SUBRAY_COMBINE_DURING_SWEEP
                indirect_tex[dispatch_vx + int3(CSGI_VOLUME_DIMS * indirect_dir_idx, 0, 0)]
                    = combined_indirect * (direct_opacity_tex[dispatch_vx] / SUBRAY_COUNT);
            #endif
        }
    }}
}

// 16 threads in a group seem fastest; cache behavior? Not enough threads to fill the GPU with larger groups?
[numthreads(4, 4, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
    #if 0
        switch (dispatch_thread_id.z) {
            case 0: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 0, __cascade_idx); break;
            case 1: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 1, __cascade_idx); break;
            case 2: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 2, __cascade_idx); break;
            case 3: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 3, __cascade_idx); break;
            case 4: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 4, __cascade_idx); break;
            case 5: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 5, __cascade_idx); break;
            case 6: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 6, __cascade_idx); break;
            case 7: diagonal_sweep_volume(dispatch_thread_id.xy, CSGI_CARDINAL_DIRECTION_COUNT + 7, __cascade_idx); break;
        }
    #else
        const uint indirect_dir_idx = CSGI_CARDINAL_DIRECTION_COUNT + dispatch_thread_id.z;
        if (__cascade_idx == CSGI_CASCADE_COUNT - 1) {
            // Fast path; no extra cascade sampling needed
            diagonal_sweep_volume(dispatch_thread_id.xy, indirect_dir_idx, CSGI_CASCADE_COUNT - 1);
        } else {
            diagonal_sweep_volume(dispatch_thread_id.xy, indirect_dir_idx, __cascade_idx);
        }
    #endif
}

