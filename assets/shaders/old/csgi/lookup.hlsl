#include "../inc/math_const.hlsl"

// #define CSGI_SUPPORT_DIRECT_LIGHT_LOOKUP 1

float henyey_greenstein(float g, float costh) {
    return (1.0 - g * g) / max(1e-5, 4.0 * M_PI * pow(max(0.0, 1.0 + g * g - 2.0 * g * costh), 3.0 / 2.0));
}

struct CsgiLookupParams {
    bool use_linear_fetch;
    bool use_bent_normal;
    bool sample_directional_radiance;
    bool sample_phase;
    float phase_g;
    bool sample_specular;
    float3 directional_radiance_direction;
    float directional_radiance_phong_exponent;
    float3 bent_normal;
    float max_normal_offset_scale;
    int debug_single_direction;

    #if CSGI_SUPPORT_DIRECT_LIGHT_LOOKUP
        bool direct_light_only;
    #endif

    static CsgiLookupParams make_default() {
        CsgiLookupParams res;
        res.use_linear_fetch = true;
        res.use_bent_normal = false;
        res.bent_normal = 0;
        res.sample_directional_radiance = false;
        res.sample_phase = false;
        res.phase_g = 0;
        res.sample_specular = false;
        res.directional_radiance_phong_exponent = 50.0;
        res.max_normal_offset_scale = 2.0;
        res.debug_single_direction = -1;

        #if CSGI_SUPPORT_DIRECT_LIGHT_LOOKUP
            res.direct_light_only = false;
        #endif

        return res;
    }

    CsgiLookupParams with_bent_normal(float3 v) {
        CsgiLookupParams res = this;
        res.use_bent_normal = true;
        res.bent_normal = v;
        return res;
    }

    CsgiLookupParams with_linear_fetch(bool v) {
        CsgiLookupParams res = this;
        res.use_linear_fetch = v;
        return res;
    }

    CsgiLookupParams with_sample_directional_radiance(float3 v) {
        CsgiLookupParams res = this;
        res.sample_directional_radiance = true;
        res.directional_radiance_direction = v;
        return res;
    }

    CsgiLookupParams with_sample_phase(float g, float3 dir) {
        CsgiLookupParams res = this;
        res.sample_phase = true;
        res.directional_radiance_direction = dir;
        res.phase_g = g;
        return res;
    }

    CsgiLookupParams with_sample_specular(float3 v) {
        CsgiLookupParams res = this;
        res.sample_specular = true;
        res.directional_radiance_direction = v;
        return res;
    }

    CsgiLookupParams with_directional_radiance_phong_exponent(float v) {
        CsgiLookupParams res = this;
        res.directional_radiance_phong_exponent = v;
        return res;
    }

    CsgiLookupParams with_max_normal_offset_scale(float v) {
        CsgiLookupParams res = this;
        res.max_normal_offset_scale = v;
        return res;
    }

    CsgiLookupParams with_debug_single_direction(uint v) {
        CsgiLookupParams res = this;
        res.debug_single_direction = v;
        return res;
    }

#if CSGI_SUPPORT_DIRECT_LIGHT_LOOKUP
    CsgiLookupParams with_direct_light_only(bool v) {
        CsgiLookupParams res = this;
        res.direct_light_only = true;
        return res;
    }
#endif
};


float3 lookup_csgi(float3 pos, float3 normal, CsgiLookupParams params) {
    // Shift to grid-local coords
    pos -= CSGI_VOLUME_ORIGIN;

    const float normal_offset_scale = min(
        params.use_linear_fetch ? 1.51 : 1.01,
        params.max_normal_offset_scale
    );

    uint cascade_idx;
    float3 shifted_pos = pos;

    // TODO: carefully evaluate where a lookup can be kicked out of a volume, and optimize this.
    // currently needs three iterations to resolve all edge cases.
    [unroll]
    for (uint cascade_search_iter = 0; cascade_search_iter < (CSGI_CASCADE_COUNT > 1 ? 3 : 1); ++cascade_search_iter) {
        cascade_idx = csgi_cascade_idx_for_pos(shifted_pos + CSGI_VOLUME_ORIGIN);

        float3 candidate_shifted_pos = pos;

        // Normal bias
        #if CSGI_SUPPORT_DIRECT_LIGHT_LOOKUP
            if (params.direct_light_only) {
                candidate_shifted_pos += (normal * 1e-3) * csgi_voxel_size(cascade_idx);
            } else
        #endif
        {
            candidate_shifted_pos += (normal * normal_offset_scale) * csgi_voxel_size(cascade_idx);        
        }

        if (params.use_bent_normal) {
            float3 to_eye = get_eye_position() - pos;

            for (int gi_slice_idx = 0; gi_slice_idx < CSGI_CARDINAL_DIRECTION_COUNT; ++gi_slice_idx) {
                const float3 slice_dir = CSGI_DIRECT_DIRS[gi_slice_idx];

                // Already normal-biased; only shift in the tangent plane.
                const float3 offset_dir = slice_dir - normal * dot(normal, slice_dir);

                if (params.use_linear_fetch) {
                    candidate_shifted_pos += 1.0 * offset_dir * clamp(3 * dot(slice_dir, params.bent_normal), 0.0, 0.5) * csgi_voxel_size(cascade_idx);
                }
            }
        }

        shifted_pos = candidate_shifted_pos;
    }
    const float3 vol_pos = shifted_pos;

    float3 total_gi = 0;
    float total_gi_wt = 0;

    int3 gi_vx = int3(floor(vol_pos / csgi_voxel_size(cascade_idx)));
    if (gi_volume_contains_vx(frame_constants.gi_cascades[cascade_idx], gi_vx)) {
        gi_vx -= frame_constants.gi_cascades[cascade_idx].scroll_int.xyz;
        gi_vx = csgi_wrap_vx_within_cascade(gi_vx);

    #if CSGI_SUPPORT_DIRECT_LIGHT_LOOKUP
        if (params.direct_light_only) {
            for (uint gi_slice_idx = 0; gi_slice_idx < CSGI_CARDINAL_DIRECTION_COUNT; ++gi_slice_idx) {
                const float3 slice_dir = float3(CSGI_DIRECT_DIRS[gi_slice_idx]);
                float wt = saturate(dot(normalize(-slice_dir), normal));
                float4 radiance_alpha = csgi_direct_tex[gi_vx + int3(CSGI_VOLUME_DIMS * gi_slice_idx, 0, 0)];
                total_gi += radiance_alpha.rgb / max(1e-5, radiance_alpha.a) * wt;
                total_gi_wt += wt;
            }
        } else
    #endif
        {
            for (uint gi_slice_idx = 0; gi_slice_idx < CSGI_TOTAL_DIRECTION_COUNT; ++gi_slice_idx) {
                if (params.debug_single_direction != -1) {
                    if (gi_slice_idx != params.debug_single_direction) {
                        continue;
                    }
                }

                const float3 slice_dir = float3(CSGI_INDIRECT_DIRS[gi_slice_idx]);
                float wt;

                const float wrap_around_bias = 0.2;

                if (params.sample_directional_radiance) {
                    wt = saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), params.directional_radiance_direction));
                    wt = pow(wt, params.directional_radiance_phong_exponent);
                } else if (params.sample_specular && params.directional_radiance_phong_exponent > 0.1) {
                    wt = saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), params.directional_radiance_direction));
                    wt = pow(wt, params.directional_radiance_phong_exponent);
                    wt *= saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), normal));
                } else if (params.sample_phase) {
                    float cos_theta = dot(wrap_around_bias + (1.0 - wrap_around_bias) * normalize(slice_dir), params.directional_radiance_direction);
                    wt = henyey_greenstein(params.phase_g, cos_theta);
                } else {
                    wt = saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), normal));
                }

                //wt = normalize(slice_dir).x > 0.99 ? 1.0 : 0.0;
                //wt *= wt;
                
                float3 gi_uv = frac(
                    (float3(gi_vx) + (vol_pos / csgi_voxel_size(cascade_idx) - float3(gi_vx))) / CSGI_VOLUME_DIMS
                );

                // Discontinuous UV == incorrect sampling; would need manual bilinear, but maybe this is enough
                const bool linear_fetch_valid = all(abs(gi_uv - 0.5) < 0.5 - 0.5 / CSGI_VOLUME_DIMS);

                if (params.use_linear_fetch && linear_fetch_valid) {
                    gi_uv = clamp(gi_uv, 0.5 / CSGI_VOLUME_DIMS, 1.0 - (0.5 / CSGI_VOLUME_DIMS));
                    gi_uv.x /= CSGI_TOTAL_DIRECTION_COUNT;
                    gi_uv.x += float(gi_slice_idx) / CSGI_TOTAL_DIRECTION_COUNT;
                    
                    total_gi += csgi_indirect_tex
                        [NonUniformResourceIndex(cascade_idx)]
                        .SampleLevel(sampler_lnc, gi_uv, 0).rgb * wt;

                    total_gi_wt += wt;
                } else {
                    total_gi += csgi_indirect_tex
                        [NonUniformResourceIndex(cascade_idx)]
                        [gi_vx + int3(CSGI_VOLUME_DIMS * gi_slice_idx, 0, 0)].rgb * wt;

                    total_gi_wt += wt;
                }
            }
        }
    } else {
    #ifndef CSGI_LOOKUP_NO_DIRECT
        // TODO: reduce copy-pasta
        for (uint gi_slice_idx = 0; gi_slice_idx < CSGI_TOTAL_DIRECTION_COUNT; ++gi_slice_idx) {
            if (params.debug_single_direction != -1) {
                if (gi_slice_idx != params.debug_single_direction) {
                    continue;
                }
            }

            const float3 slice_dir = float3(CSGI_INDIRECT_DIRS[gi_slice_idx]);
            float wt;

            const float wrap_around_bias = 0.2;

            if (params.sample_directional_radiance) {
                wt = saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), params.directional_radiance_direction));
                wt = pow(wt, params.directional_radiance_phong_exponent);
            } else if (params.sample_specular && params.directional_radiance_phong_exponent > 0.1) {
                wt = saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), params.directional_radiance_direction));
                wt = pow(wt, params.directional_radiance_phong_exponent);
                wt *= saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), normal));
            } else if (params.sample_phase) {
                float cos_theta = dot(wrap_around_bias + (1.0 - wrap_around_bias) * normalize(slice_dir), params.directional_radiance_direction);
                wt = henyey_greenstein(params.phase_g, cos_theta);
            } else {
                wt = saturate(wrap_around_bias + (1.0 - wrap_around_bias) * dot(normalize(slice_dir), normal));
            }

            const float3 cube_sample_in_dir = sky_cube_tex.SampleLevel(sampler_llr, slice_dir, 0).rgb;
            total_gi += cube_sample_in_dir * wt;
            total_gi_wt += wt;
        }
    #endif
    }

    //total_gi_wt = 1;
    //total_gi = float3(uint3(gi_vx) % 4) / 4;

    return total_gi / max(1e-20, total_gi_wt);
}
