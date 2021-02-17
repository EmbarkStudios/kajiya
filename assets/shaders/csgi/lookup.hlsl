struct CsgiLookupParams {
    bool use_grid_linear_fetch;
    int debug_slice_idx;
    float normal_cutoff;

    static CsgiLookupParams make_default() {
        CsgiLookupParams res;
        res.use_grid_linear_fetch = true;
        res.debug_slice_idx = -1;
        res.normal_cutoff = 1e-3;
        return res;
    }
};

uint closest_csgi_dir(float3 dir) {
    float best_dot = -1;
    uint best_idx = 0;

    for (uint i = 0; i < GI_SLICE_COUNT; ++i) {
        const float3 slice_dir = SLICE_DIRS[i].xyz;
        const float d = dot(slice_dir, dir);
        if (d > best_dot) {
            best_dot = d;
            best_idx = i;
        }
    }

    return best_idx;
}

float3 lookup_csgi(float3 pos, float3 normal, CsgiLookupParams params) {
    float3 irradiance = 0.0.xxx;
    float wsum = 0.0;

    for (uint gi_slice_iter = 0; gi_slice_iter < GI_SLICE_COUNT; ++gi_slice_iter) {
    //const uint gi_slice_iter = closest_csgi_dir(-normal); [unroll] for (uint dummy = 0; dummy < 1; ++dummy) {
        const uint gi_slice_idx = params.debug_slice_idx == -1 ? gi_slice_iter : params.debug_slice_idx;

        const float3 slice_dir = SLICE_DIRS[gi_slice_idx].xyz;
        const float ndotl = -dot(slice_dir, normal);

        if (ndotl < params.normal_cutoff) {
            continue;
        }

        const float visibility = max(0.0, ndotl);
        const float normal_offset_scale = min(1.5, 1.1 / ndotl);
        //const float normal_offset_scale = 1.01;
        //const float normal_offset_scale = 1.5;

        const float3x3 slice_rot = build_orthonormal_basis(slice_dir);
        const float3 volume_center = SLICE_CENTERS[gi_slice_iter].xyz;
        //const float3 volume_center = gi_volume_center(slice_rot);

        const float3 vol_pos = (pos - volume_center + normal * normal_offset_scale * GI_VOXEL_SIZE);

        int3 gi_vx = int3(mul(vol_pos, slice_rot) / GI_VOXEL_SIZE + GI_VOLUME_DIMS / 2);
        {
            float3 radiance = 0;

        #ifndef CSGI_LOOKUP_NEAREST_ONLY
            if (params.use_grid_linear_fetch) {
                // HACK: if a hit is encountered early in a cell, the entire cell will report seeing a surface,
                // even though most of it could be shadowed. This shifts the lookup to be deeper inside the surface.
                //float3 depth_bias = float3(0, 0, -1.0 / GI_VOLUME_DIMS);
                float3 depth_bias = 0;

                float3 gi_uv = mul((vol_pos / GI_VOXEL_SIZE / (GI_VOLUME_DIMS / 2)), slice_rot) * 0.5 + 0.5 + depth_bias;

                if (all(gi_uv == saturate(gi_uv))) {
                    gi_uv = clamp(gi_uv, 0.5 / GI_VOLUME_DIMS, 1.0 - (0.5 / GI_VOLUME_DIMS));
                    gi_uv.x /= GI_SLICE_COUNT;
                    gi_uv.x += float(gi_slice_idx) / GI_SLICE_COUNT;
                    radiance = cascade0_tex.SampleLevel(sampler_lnc, gi_uv, 0).rgb;
                }
            } else
        #endif
            {
                // Nearest lookup

                //const int3 depth_bias = int3(0, 0, -1);
                const int3 depth_bias = int3(0, 0, 0);

                if (gi_vx.x >= 0 && gi_vx.x < GI_VOLUME_DIMS) {
                    radiance = cascade0_tex[gi_vx + depth_bias + int3(GI_VOLUME_DIMS * gi_slice_idx, 0, 0)].rgb;
                }
            }

            irradiance += radiance * visibility;
            wsum += visibility;
        }
    }

    return irradiance / max(1e-5, wsum);
}
