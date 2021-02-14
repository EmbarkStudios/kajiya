struct CsgiLookupParams {
    bool use_grid_linear_fetch;
    bool use_pretrace;
    int debug_slice_idx;

    //float4 slice_dirs[GI_SLICE_COUNT];

    //Texture3D<float4> cascade0_tex;
    //Texture3D<float4> alt_cascade0_tex;
};

float3 lookup_csgi(float3 pos, float3 normal, CsgiLookupParams params) {
    const float normal_offset_scale = 1.1;
    const float3 vol_pos = (pos - GI_VOLUME_CENTER + normal * normal_offset_scale * (GI_VOLUME_SIZE / GI_VOLUME_DIMS));

    float3 irradiance = 0.0.xxx;

    for (uint gi_slice_iter = 0; gi_slice_iter < GI_SLICE_COUNT; ++gi_slice_iter) {
        const uint gi_slice_idx = params.debug_slice_idx == -1 ? gi_slice_iter : params.debug_slice_idx;

        const float3x3 slice_rot = build_orthonormal_basis(SLICE_DIRS[gi_slice_idx].xyz);

        int3 gi_vx = int3(mul(vol_pos / (GI_VOLUME_SIZE / GI_VOLUME_DIMS), slice_rot) + GI_VOLUME_DIMS / 2);
        {
            float3 radiance = 0;

            if (params.use_grid_linear_fetch) {
                float3 gi_uv = mul((vol_pos / (GI_VOLUME_SIZE / GI_VOLUME_DIMS) / (GI_VOLUME_DIMS / 2)), slice_rot) * 0.5 + 0.5;

                if (all(gi_uv == saturate(gi_uv))) {
                    gi_uv = clamp(gi_uv, 0.5 / GI_VOLUME_DIMS, 1.0 - (0.5 / GI_VOLUME_DIMS));
                    gi_uv.x /= GI_SLICE_COUNT;
                    gi_uv.x += float(gi_slice_idx) / GI_SLICE_COUNT;

                    if (params.use_pretrace) {
                        radiance = alt_cascade0_tex.SampleLevel(sampler_lnc, gi_uv, 0).rgb;
                    } else {
                        radiance = cascade0_tex.SampleLevel(sampler_lnc, gi_uv, 0).rgb;
                    }
                }
            } else {
                // Nearest lookup

                if (gi_vx.x >= 0 && gi_vx.x < GI_VOLUME_DIMS) {
                    if (params.use_pretrace) {
                        radiance = alt_cascade0_tex[gi_vx + int3(GI_VOLUME_DIMS * gi_slice_idx, 0, 0)].rgb;
                    } else {
                        radiance = cascade0_tex[gi_vx + int3(GI_VOLUME_DIMS * gi_slice_idx, 0, 0)].rgb;
                    }
                }
            }

            const float3 ldir = SLICE_DIRS[gi_slice_idx].xyz;
            const float visibility = saturate(0.2 + dot(mul(ldir, normal), float3(0, 0, -1)));

            irradiance += radiance * visibility;
        }
    }

    return irradiance * M_PI / GI_SLICE_COUNT;
}
