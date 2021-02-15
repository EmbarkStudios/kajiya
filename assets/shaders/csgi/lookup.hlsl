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

    float3 irradiance = 0.0.xxx;

    for (uint gi_slice_iter = 0; gi_slice_iter < GI_SLICE_COUNT; ++gi_slice_iter) {
        const uint gi_slice_idx = params.debug_slice_idx == -1 ? gi_slice_iter : params.debug_slice_idx;

        const float3x3 slice_rot = build_orthonormal_basis(SLICE_DIRS[gi_slice_idx].xyz);
        const float3 vol_pos = (pos - gi_volume_center(slice_rot) + normal * normal_offset_scale * GI_VOXEL_SIZE);

        int3 gi_vx = int3(mul(vol_pos, slice_rot) / GI_VOXEL_SIZE + GI_VOLUME_DIMS / 2);
        {
            float3 radiance = 0;

        #ifndef CSGI_LOOKUP_NEAREST_ONLY
            if (params.use_grid_linear_fetch) {
                float3 gi_uv = mul((vol_pos / GI_VOXEL_SIZE / (GI_VOLUME_DIMS / 2)), slice_rot) * 0.5 + 0.5;

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
            } else
        #endif
            {
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
            const float ndotl = dot(mul(ldir, normal), float3(0, 0, -1));

            // TODO: remove or justify hack (bias to make NdotL less harsh for a cone of directions)
            //const float visibility = max(0.0, ndotl);
            const float visibility = max(0.0, lerp(ndotl, 0.9, 0.05));

            irradiance += radiance * visibility;
        }
    }

    return irradiance * M_PI / GI_SLICE_COUNT;
}
