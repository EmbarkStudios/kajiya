struct OcclusionScreenRayMarch {
    uint max_sample_count;
    float2 raymarch_start_uv;
    float3 raymarch_start_cs;
    float3 raymarch_start_ws;
    float3 raymarch_end_ws;
    float2 fullres_depth_tex_size;

    bool use_halfres_depth;
    float2 halfres_depth_tex_size;
    Texture2D<float> halfres_depth_tex;

    Texture2D<float> fullres_depth_tex;

    bool use_color_bounce;
    Texture2D<float3> fullres_color_bounce_tex;

    OcclusionScreenRayMarch with_color_bounce(Texture2D<float3> _fullres_color_bounce_tex) {
        OcclusionScreenRayMarch res = this;
        res.use_color_bounce = true;
        res.fullres_color_bounce_tex = _fullres_color_bounce_tex;
        return res;
    }

    OcclusionScreenRayMarch with_max_sample_count(uint _max_sample_count) {
        OcclusionScreenRayMarch res = this;
        res.max_sample_count = _max_sample_count;
        return res;
    }

    OcclusionScreenRayMarch with_halfres_depth(
        float2 _halfres_depth_tex_size,
        Texture2D<float> _halfres_depth_tex
    ) {
        OcclusionScreenRayMarch res = this;
        res.use_halfres_depth = true;
        res.halfres_depth_tex_size = _halfres_depth_tex_size;
        res.halfres_depth_tex = _halfres_depth_tex;
        return res;
    }

    OcclusionScreenRayMarch with_fullres_depth(
        Texture2D<float> _fullres_depth_tex
    ) {
        OcclusionScreenRayMarch res = this;
        res.use_halfres_depth = false;
        res.fullres_depth_tex = _fullres_depth_tex;
        return res;
    }

    static OcclusionScreenRayMarch create(
        float2 raymarch_start_uv, float3 raymarch_start_cs, float3 raymarch_start_ws,
        float3 raymarch_end_ws,
        float2 fullres_depth_tex_size
    ) {
        OcclusionScreenRayMarch res;
        res.max_sample_count = 4;
        res.raymarch_start_uv = raymarch_start_uv;
        res.raymarch_start_cs = raymarch_start_cs;
        res.raymarch_start_ws = raymarch_start_ws;
        res.raymarch_end_ws = raymarch_end_ws;

        res.fullres_depth_tex_size = fullres_depth_tex_size;

        res.use_color_bounce = false;
        return res;
    }

    void march(
        inout float visibility,
        inout float3 sample_radiance
    ) {
        const float2 raymarch_end_uv = cs_to_uv(position_world_to_clip(raymarch_end_ws).xy);
        const float2 raymarch_uv_delta = raymarch_end_uv - raymarch_start_uv;
        const float2 raymarch_len_px = raymarch_uv_delta * select(use_halfres_depth, halfres_depth_tex_size, fullres_depth_tex_size);

        const uint MIN_PX_PER_STEP = 2;

        const int k_count = min(max_sample_count, int(floor(length(raymarch_len_px) / MIN_PX_PER_STEP)));

        // Depth values only have the front; assume a certain thickness.
        const float Z_LAYER_THICKNESS = 0.05;

        //const float3 raymarch_start_cs = view_ray_context.ray_hit_cs.xyz;
        const float3 raymarch_end_cs = position_world_to_clip(raymarch_end_ws).xyz;
        const float depth_step_per_px = (raymarch_end_cs.z - raymarch_start_cs.z) / length(raymarch_len_px);
        const float depth_step_per_z = (raymarch_end_cs.z - raymarch_start_cs.z) / length(raymarch_end_cs.xy - raymarch_start_cs.xy);

        float t_step = 1.0 / k_count;
        float t = 0.5 * t_step;
        for (int k = 0; k < k_count; ++k) {
            const float3 interp_pos_cs = lerp(raymarch_start_cs, raymarch_end_cs, t);

            // The point-sampled UV could end up with a quite different depth value
            // than the one interpolated along the ray (which is not quantized).
            // This finds a conservative bias for the comparison.
            const float2 uv_at_interp = cs_to_uv(interp_pos_cs.xy);

            uint2 px_at_interp;
            float depth_at_interp;

            if (use_halfres_depth) {
                px_at_interp = (uint2(floor(uv_at_interp * fullres_depth_tex_size - HALFRES_SUBSAMPLE_OFFSET)) & ~1u) + HALFRES_SUBSAMPLE_OFFSET;
                depth_at_interp = halfres_depth_tex[px_at_interp >> 1u];
            } else {
                px_at_interp = floor(uv_at_interp * fullres_depth_tex_size);
                depth_at_interp = fullres_depth_tex[px_at_interp];
            }

            const float2 quantized_cs_at_interp = uv_to_cs((px_at_interp + 0.5) / fullres_depth_tex_size);

            const float biased_interp_z = raymarch_start_cs.z + depth_step_per_z * length(quantized_cs_at_interp - raymarch_start_cs.xy);

            if (depth_at_interp > biased_interp_z) {
                const float depth_diff = inverse_depth_relative_diff(interp_pos_cs.z, depth_at_interp);

                float hit = smoothstep(
                    Z_LAYER_THICKNESS,
                    Z_LAYER_THICKNESS * 0.5,
                    depth_diff);

                if (RTDGI_RESTIR_SPATIAL_USE_RAYMARCH_COLOR_BOUNCE) {
                    const float3 hit_radiance = fullres_color_bounce_tex.SampleLevel(sampler_llc, cs_to_uv(interp_pos_cs.xy), 0).rgb;
                    const float3 prev_sample_radiance = sample_radiance;
                    
                    sample_radiance = lerp(sample_radiance, hit_radiance, hit);

                    // Heuristic: don't allow getting _brighter_ from accidental
                    // hits reused from neighbors. This can cause some darkening,
                    // but also fixes reduces noise (expecting to hit dark, hitting bright),
                    // and improves a few cases that otherwise look unshadowed.
                    visibility *= min(1.0, sRGB_to_luminance(prev_sample_radiance) / sRGB_to_luminance(sample_radiance));
                } else {
                    visibility *= 1 - hit;
                }

                if (depth_diff > Z_LAYER_THICKNESS) {
                    // Going behind an object; could be sketchy.
                    // Note: maybe nuke.. causes bias around foreground objects.
                    //relevance *= 0.2;
                }
            }

            t += t_step;
        }
    }
};
