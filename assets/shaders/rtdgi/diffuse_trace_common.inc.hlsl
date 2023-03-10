// Should be 1, but rarely matters for the diffuse bounce, so might as well save a few cycles.
#define USE_SOFT_SHADOWS 0

#define USE_IRCACHE 1
#define USE_WORLD_RADIANCE_CACHE 0

#define ROUGHNESS_BIAS 0.5
#define USE_SCREEN_GI_REPROJECTION 1
#define USE_SWIZZLE_TILE_PIXELS 0

#define USE_EMISSIVE 1
#define USE_LIGHTS 1

#define USE_SKY_CUBE_TEX 1

static const float SKY_DIST = 1e4;

float3 sample_environment_light(float3 dir) {
    #if USE_SKY_CUBE_TEX
        return sky_cube_tex.SampleLevel(sampler_llr, dir, 0).rgb;
    #else
        return atmosphere_default(dir, SUN_DIRECTION);
    #endif
}

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    float3 out_value;
    float3 hit_normal_ws;
    float hit_t;
    float pdf;
    bool is_hit;
};

TraceResult do_the_thing(uint2 px, float3 normal_ws, inout uint rng, RayDesc outgoing_ray) {
    float3 total_radiance = 0.0.xxx;
    float3 hit_normal_ws = -outgoing_ray.Direction;

    #if USE_WORLD_RADIANCE_CACHE
        WrcFarField far_field =
            WrcFarFieldQuery::from_ray(outgoing_ray.Origin, outgoing_ray.Direction)
                .with_interpolation_urand(float3(
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng))
                ))
                .with_query_normal(normal_ws)
                .query();
    #else
        WrcFarField far_field = WrcFarField::create_miss();
    #endif

    if (far_field.is_hit()) {
        outgoing_ray.TMax = far_field.probe_t;
    }

    float hit_t = outgoing_ray.TMax;

    // cosine-weighted
    //float pdf = 1.0 / M_PI;

    // uniform
    float pdf = max(0.0, 1.0 / (dot(normal_ws, outgoing_ray.Direction) * 2 * M_PI));

    const float reflected_cone_spread_angle = 0.03;
    const RayCone ray_cone =
        pixel_ray_cone_from_image_height(gbuffer_tex_size.y * 0.5)
        .propagate(reflected_cone_spread_angle, length(outgoing_ray.Origin - get_eye_position()));

    const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
        .with_cone(ray_cone)
        .with_cull_back_faces(false)
        .with_path_length(1)
        .trace(acceleration_structure);

    if (primary_hit.is_hit) {
        hit_t = primary_hit.ray_t;
        GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
        hit_normal_ws = gbuffer.normal;

        // Project the sample into clip space, and check if it's on-screen
        const float3 primary_hit_cs = position_world_to_sample(primary_hit.position);
        const float2 primary_hit_uv = cs_to_uv(primary_hit_cs.xy);
        const float primary_hit_screen_depth = depth_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0);
        //const GbufferDataPacked primary_hit_screen_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[int2(primary_hit_uv * gbuffer_tex_size.xy)]));
        //const float3 primary_hit_screen_normal_ws = primary_hit_screen_gbuffer.unpack_normal();
        bool is_on_screen = true
            && all(abs(primary_hit_cs.xy) < 1.0)
            && inverse_depth_relative_diff(primary_hit_cs.z, primary_hit_screen_depth) < 5e-3
            // TODO
            //&& dot(primary_hit_screen_normal_ws, -outgoing_ray.Direction) > 0.0
            //&& dot(primary_hit_screen_normal_ws, gbuffer.normal) > 0.7
            ;

        // If it is on-screen, we'll try to use its reprojected radiance from the previous frame
        float4 reprojected_radiance = 0;
        if (is_on_screen) {
            reprojected_radiance =
                reprojected_gi_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0)
                * frame_constants.pre_exposure_delta;

            // Check if the temporal reprojection is valid.
            is_on_screen = reprojected_radiance.w > 0;
        }

        gbuffer.roughness = lerp(gbuffer.roughness, 1.0, ROUGHNESS_BIAS);
        const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
        const float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);
        const LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

        // Sun
        float3 sun_radiance = SUN_COLOR;
        if (any(sun_radiance) > 0) {
            const float3 to_light_norm = sample_sun_direction(
                blue_noise_for_pixel(px, rng).xy,
                USE_SOFT_SHADOWS
            );

            const bool is_shadowed =
                rt_is_shadowed(
                    acceleration_structure,
                    new_ray(
                        primary_hit.position,
                        to_light_norm,
                        1e-4,
                        SKY_DIST
                ));

            const float3 wi = mul(to_light_norm, tangent_to_world);
            const float3 brdf_value = brdf.evaluate(wo, wi) * max(0.0, wi.z);
            const float3 light_radiance = select(is_shadowed, 0.0, sun_radiance);
            total_radiance += brdf_value * light_radiance;
        }

        if (USE_EMISSIVE) {
            total_radiance += gbuffer.emissive;
        }

        if (USE_SCREEN_GI_REPROJECTION && is_on_screen) {
            total_radiance += reprojected_radiance.rgb * gbuffer.albedo;
        } else {
            if (USE_LIGHTS) {
                float2 urand = float2(
                    uint_to_u01_float(hash1_mut(rng)),
                    uint_to_u01_float(hash1_mut(rng))
                );

                for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1) {
                    TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
                    LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
                    const float3 shadow_ray_origin = primary_hit.position;
                    const float3 to_light_ws = light_sample.pos - shadow_ray_origin;
                    const float dist_to_light2 = dot(to_light_ws, to_light_ws);
                    const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

                    const float to_psa_metric =
                        max(0.0, dot(to_light_norm_ws, gbuffer.normal))
                        * max(0.0, dot(to_light_norm_ws, -light_sample.normal))
                        / dist_to_light2;

                    if (to_psa_metric > 0.0) {
                        const bool is_shadowed =
                            rt_is_shadowed(
                                acceleration_structure,
                                new_ray(
                                    shadow_ray_origin,
                                    to_light_norm_ws,
                                    1e-3,
                                    sqrt(dist_to_light2) - 2e-3
                            ));

                        #if 1
                            const float3 bounce_albedo = lerp(gbuffer.albedo, 1.0.xxx, 0.04);
                            const float3 brdf_value = bounce_albedo * to_psa_metric / M_PI;
                        #else
                            const float3 wi = mul(to_light_norm_ws, tangent_to_world);
                            const float3 brdf_value = brdf.evaluate(wo, wi) * to_psa_metric;
                        #endif

                        total_radiance +=
                            select(!is_shadowed, (triangle_light.radiance() * brdf_value / light_sample.pdf.value), 0);
                    }
                }
            }

            if (USE_IRCACHE) {
                const float3 gi = IrcacheLookupParams::create(
                    outgoing_ray.Origin,
                    primary_hit.position,
                    gbuffer.normal)
                    .with_query_rank(1)
                    .lookup(rng);

                total_radiance += gi * gbuffer.albedo;
            }
        }
    } else {
        if (far_field.is_hit()) {
            total_radiance += far_field.radiance;
            hit_t = far_field.approx_surface_t;
            pdf = 1.0 / far_field.inv_pdf;
        } else {
            total_radiance += sample_environment_light(outgoing_ray.Direction);
        }
    }

    float3 out_value = total_radiance;

    //out_value /= reservoir.p_sel;

    TraceResult result;
    result.out_value = out_value;
    result.hit_t = hit_t;
    result.hit_normal_ws = hit_normal_ws;
    result.pdf = pdf;
    result.is_hit = primary_hit.is_hit;
    return result;
}
