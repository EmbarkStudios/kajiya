float3 point_sample_csgi_subray_indirect_subray(float3 pos, uint dir_idx, uint subray) {
    const uint cascade_idx = csgi_cascade_idx_for_pos(pos);
    const float3 vol_pos = (pos - CSGI_VOLUME_ORIGIN);
    int3 gi_vx = int3(floor(vol_pos / csgi_voxel_size(cascade_idx)));

    if (!gi_volume_contains_vx(frame_constants.gi_cascades[cascade_idx], gi_vx)) {
        return 0.0;
    }

    if (dir_idx < CSGI_CARDINAL_DIRECTION_COUNT) {
        return csgi_subray_indirect_tex[cascade_idx][
            csgi_cardinal_vx_dir_subray_to_subray_vx(csgi_wrap_vx_within_cascade(gi_vx), dir_idx, subray)
        ];
    } else {
        return csgi_subray_indirect_tex[cascade_idx][
            csgi_diagonal_vx_dir_subray_to_subray_vx(csgi_wrap_vx_within_cascade(gi_vx), dir_idx, subray)
        ];
    }
}

// TODO: precompute
float3 point_sample_csgi_subray_indirect(float3 pos, float3 v) {
    uint dir_idx = 0;
    {
        float best_dot = 0;
        for (uint gi_slice_idx = 0; gi_slice_idx < CSGI_TOTAL_DIRECTION_COUNT; ++gi_slice_idx) {
            const float3 slice_dir = normalize(float3(CSGI_INDIRECT_DIRS[gi_slice_idx]));
            float d = dot(slice_dir, v);
            if (d > best_dot) {
                best_dot = d;
                dir_idx = gi_slice_idx;
            }
        }
    }

    uint subray = 0;
    if (dir_idx < CSGI_CARDINAL_DIRECTION_COUNT) {
        static const uint TANGENT_COUNT = 4;
        uint tangent_dir_indices[TANGENT_COUNT];
        {for (uint i = 0; i < TANGENT_COUNT; ++i) {
            tangent_dir_indices[i] = ((dir_idx & uint(~1)) + 2 + i) % CSGI_CARDINAL_DIRECTION_COUNT;
        }}

        static const float4 subray_weights[CSGI_CARDINAL_SUBRAY_COUNT] = CSGI_CARDINAL_SUBRAY_TANGENT_WEIGHTS;

        const float main_direction_influence = 1.0;

        float best_subray_dot = 0;
        for (uint s = 0; s < CSGI_CARDINAL_SUBRAY_COUNT; ++s) {
            float3 dir = 0;
            for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
                const uint tangent_dir_idx = tangent_dir_indices[tangent_i];
                const int3 tangent_dir = CSGI_DIRECT_DIRS[tangent_dir_idx];
                dir += normalize(tangent_dir + main_direction_influence * CSGI_DIRECT_DIRS[dir_idx]) * subray_weights[s][tangent_i];
            }

            dir = normalize(dir);
            float d = dot(dir, v);

            if (d > best_subray_dot) {
                best_subray_dot = d;
                subray = s;
            }
        }
    } else {
        const int3 indirect_dir = CSGI_INDIRECT_DIRS[dir_idx];
        const uint dir_i_idx = 0 + (indirect_dir.x > 0 ? 1 : 0);
        const uint dir_j_idx = 2 + (indirect_dir.y > 0 ? 1 : 0);
        const uint dir_k_idx = 4 + (indirect_dir.z > 0 ? 1 : 0);
        const int3 dir_i = CSGI_DIRECT_DIRS[dir_i_idx];
        const int3 dir_j = CSGI_DIRECT_DIRS[dir_j_idx];
        const int3 dir_k = CSGI_DIRECT_DIRS[dir_k_idx];

        const uint TANGENT_COUNT = 3;
        const int3 tangent_dirs[TANGENT_COUNT] = {
            dir_i, dir_j, dir_k
        };

        static const float skew = 0.5;
        static const float3 subray_wts[CSGI_DIAGONAL_SUBRAY_COUNT] = CSGI_DIAGONAL_SUBRAY_TANGENT_WEIGHTS;

        float best_subray_dot = 0;
        for (uint s = 0; s < CSGI_DIAGONAL_SUBRAY_COUNT; ++s) {
            float3 dir = 0;
            for (uint tangent_i = 0; tangent_i < TANGENT_COUNT; ++tangent_i) {
                const int3 tangent_dir = tangent_dirs[tangent_i];
                dir += normalize(tangent_dir + indirect_dir) * subray_wts[s][tangent_i];
            }

            dir = normalize(dir);
            float d = dot(dir, v);

            if (d > best_subray_dot) {
                best_subray_dot = d;
                subray = s;
            }
        }
    }

    return point_sample_csgi_subray_indirect_subray(pos, dir_idx, subray);
}
