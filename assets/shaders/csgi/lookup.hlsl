#include "../inc/math_const.hlsl"

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
    bool direct_light_only;
    float3 directional_radiance_direction;
    float directional_radiance_phong_exponent;
    float3 bent_normal;
    float max_normal_offset_scale;
    int debug_single_direction;

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
        res.direct_light_only = false;
        res.max_normal_offset_scale = 2.0;
        res.debug_single_direction = -1;
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

    CsgiLookupParams with_direct_light_only(bool v) {
        CsgiLookupParams res = this;
        res.direct_light_only = true;
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
};


float3 lookup_csgi(float3 pos, float3 normal, CsgiLookupParams params) {
    const float3 volume_center = CSGI_VOLUME_CENTER;

    const float normal_offset_scale = min(
        params.use_linear_fetch ? 1.51 : 1.01,
        params.max_normal_offset_scale
    );

    //const float normal_offset_scale = 1.01;
    float3 vol_pos = pos - volume_center;

    // Normal bias
    if (!params.direct_light_only) {
        vol_pos += (normal * normal_offset_scale) * CSGI_VOXEL_SIZE;
    } else {
        vol_pos += (normal * 1e-3) * CSGI_VOXEL_SIZE;
    }

    float3 total_gi = 0;
    float total_gi_wt = 0;

    if (params.use_bent_normal) {
        const int3 gi_vx = int3(vol_pos / CSGI_VOXEL_SIZE + CSGI_VOLUME_DIMS / 2);

        float3 to_eye = get_eye_position() - pos;

        for (int gi_slice_idx = 0; gi_slice_idx < CSGI_CARDINAL_DIRECTION_COUNT; ++gi_slice_idx) {
            const float opacity = csgi_direct_tex[gi_vx + int3(CSGI_VOLUME_DIMS * gi_slice_idx, 0, 0)].a;
            const float3 slice_dir = CSGI_DIRECT_DIRS[gi_slice_idx];

            // Already normal-biased; only shift in the tangent plane.
            const float3 offset_dir = slice_dir - normal * dot(normal, slice_dir);

            if (params.use_linear_fetch) {
                vol_pos += 1.0 * offset_dir * clamp(3 * dot(slice_dir, params.bent_normal), 0.0, 0.5) * CSGI_VOXEL_SIZE;
            }
            //total_gi_wt += opacity * 1e10;
        }
    }

    const int3 gi_vx = int3(vol_pos / CSGI_VOXEL_SIZE + CSGI_VOLUME_DIMS / 2);
    if (all(gi_vx >= 0) && all(gi_vx < CSGI_VOLUME_DIMS)) {
        if (!params.direct_light_only) {
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
                
                if (params.use_linear_fetch) {
                    float3 gi_uv = (vol_pos / CSGI_VOXEL_SIZE / (CSGI_VOLUME_DIMS / 2)) * 0.5 + 0.5;

                    if (all(gi_uv == saturate(gi_uv))) {
                        gi_uv = clamp(gi_uv, 0.5 / CSGI_VOLUME_DIMS, 1.0 - (0.5 / CSGI_VOLUME_DIMS));
                        gi_uv.x /= CSGI_TOTAL_DIRECTION_COUNT;
                        gi_uv.x += float(gi_slice_idx) / CSGI_TOTAL_DIRECTION_COUNT;
                        total_gi += csgi_indirect_tex.SampleLevel(sampler_lnc, gi_uv, 0).rgb * wt;
                        total_gi_wt += wt;
                    }
                } else {
                    total_gi += csgi_indirect_tex[gi_vx + int3(CSGI_VOLUME_DIMS * gi_slice_idx, 0, 0)].rgb * wt;
                    total_gi_wt += wt;
                }
            }
        } else {
            for (uint gi_slice_idx = 0; gi_slice_idx < CSGI_CARDINAL_DIRECTION_COUNT; ++gi_slice_idx) {
                const float3 slice_dir = float3(CSGI_DIRECT_DIRS[gi_slice_idx]);
                float wt = saturate(dot(normalize(-slice_dir), normal));
                float4 radiance_alpha = csgi_direct_tex[gi_vx + int3(CSGI_VOLUME_DIMS * gi_slice_idx, 0, 0)];
                total_gi += radiance_alpha.rgb / max(1e-5, radiance_alpha.a) * wt;
                total_gi_wt += wt;
            }
        }
    }

    return total_gi / max(1e-20, total_gi_wt);
}
