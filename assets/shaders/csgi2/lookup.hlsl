struct Csgi2LookupParams {
    bool use_linear_fetch;
    bool use_bent_normal;
    float3 bent_normal;

    static Csgi2LookupParams make_default() {
        Csgi2LookupParams res;
        res.use_linear_fetch = true;
        res.use_bent_normal = false;
        res.bent_normal = 0;
        return res;
    }

    Csgi2LookupParams with_bent_normal(float3 n) {
        Csgi2LookupParams res = this;
        res.use_bent_normal = true;
        res.bent_normal = n;
        return res;
    }

    Csgi2LookupParams with_linear_fetch(bool l) {
        Csgi2LookupParams res = this;
        res.use_linear_fetch = l;
        return res;
    }
};


float3 lookup_csgi2(float3 pos, float3 normal, Csgi2LookupParams params) {
    const float3 volume_center = CSGI2_VOLUME_CENTER;

    const float normal_offset_scale = params.use_linear_fetch ? 1.51 : 1.01;
    //const float normal_offset_scale = 1.01;
    float3 vol_pos = pos - volume_center;

    // Normal bias
    vol_pos += (normal * normal_offset_scale) * CSGI2_VOXEL_SIZE;

    float3 total_gi = 0;
    float total_gi_wt = 0;

    if (params.use_bent_normal) {
        const int3 gi_vx = int3(vol_pos / CSGI2_VOXEL_SIZE + CSGI2_VOLUME_DIMS / 2);

        float3 to_eye = get_eye_position() - pos;

        for (int gi_slice_idx = 0; gi_slice_idx < CSGI2_SLICE_COUNT; ++gi_slice_idx) {
            const float opacity = csgi2_direct_tex[gi_vx + int3(CSGI2_VOLUME_DIMS * gi_slice_idx, 0, 0)].a;
            const float3 slice_dir = CSGI2_SLICE_DIRS[gi_slice_idx];

            // Already normal-biased; only shift in the tangent plane.
            const float3 offset_dir = slice_dir - normal * dot(normal, slice_dir);

            if (params.use_linear_fetch) {
                vol_pos += 1.0 * offset_dir * clamp(3 * dot(slice_dir, params.bent_normal), 0.0, 0.5) * CSGI2_VOXEL_SIZE;
            }
            //total_gi_wt += opacity * 1e10;
        }
    }

    const int3 gi_vx = int3(vol_pos / CSGI2_VOXEL_SIZE + CSGI2_VOLUME_DIMS / 2);
    if (all(gi_vx >= 0) && all(gi_vx < CSGI2_VOLUME_DIMS)) {
        //const uint gi_slice_idx = 0; {
        for (uint gi_slice_idx = 0; gi_slice_idx < CSGI2_INDIRECT_COUNT; ++gi_slice_idx) {
        //for (uint gi_slice_idx = 0; gi_slice_idx < 6; ++gi_slice_idx) {
            const float3 slice_dir = float3(CSGI2_INDIRECT_DIRS[gi_slice_idx]);
            float wt = saturate(dot(normalize(slice_dir), normal));
            //wt *= wt;
            
            if (params.use_linear_fetch) {
                float3 gi_uv = (vol_pos / CSGI2_VOXEL_SIZE / (CSGI2_VOLUME_DIMS / 2)) * 0.5 + 0.5;

                if (all(gi_uv == saturate(gi_uv))) {
                    gi_uv = clamp(gi_uv, 0.5 / CSGI2_VOLUME_DIMS, 1.0 - (0.5 / CSGI2_VOLUME_DIMS));
                    gi_uv.x /= CSGI2_INDIRECT_COUNT;
                    gi_uv.x += float(gi_slice_idx) / CSGI2_INDIRECT_COUNT;
                    total_gi += csgi2_indirect_tex.SampleLevel(sampler_lnc, gi_uv, 0).rgb * wt;
                    total_gi_wt += wt;
                }
            } else {
                total_gi += csgi2_indirect_tex[gi_vx + int3(CSGI2_VOLUME_DIMS * gi_slice_idx, 0, 0)].rgb * wt;
                total_gi_wt += wt;
            }
        }
    }

    return total_gi / max(1.0, total_gi_wt);
}
