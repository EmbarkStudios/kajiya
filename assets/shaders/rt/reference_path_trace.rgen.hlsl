#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/bindless_textures.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] RWTexture2D<float4> output_tex;

static const uint MAX_PATH_LENGTH = 30;
static const uint RUSSIAN_ROULETTE_START_PATH_LENGTH = 3;
static const float MAX_RAY_LENGTH = FLT_MAX;
//static const float MAX_RAY_LENGTH = 5.0;

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = !true;
static const bool FURNACE_TEST = false;
static const bool FURNACE_TEST_EXCLUDE_DIFFUSE = !true;
static const bool USE_PIXEL_FILTER = true;
static const bool INDIRECT_ONLY = true;
static const bool ONLY_SPECULAR_FIRST_BOUNCE = !true;
static const bool GREY_ALBEDO_FIRST_BOUNCE = !true;

float3 sample_environment_light(float3 dir) {
    //return 0.5.xxx;

    if (FURNACE_TEST) {
        return 0.5.xxx;
    }

    return atmosphere_default(dir, SUN_DIRECTION);

    float3 col = (dir.zyx * float3(1, 1, -1) * 0.5 + float3(0.6, 0.5, 0.5)) * 0.75;
    col = lerp(col, 1.3.xxx * calculate_luma(col), smoothstep(-0.2, 1.0, dir.y).xxx);
    return col;
}


// Approximate Gaussian remap
// https://www.shadertoy.com/view/MlVSzw
float inv_error_function(float x, float truncation) {
    static const float ALPHA = 0.14;
    static const float INV_ALPHA = 1.0 / ALPHA;
    static const float K = 2.0 / (M_PI * ALPHA);

	float y = log(max(truncation, 1.0 - x*x));
	float z = K + 0.5 * y;
	return sqrt(max(0.0, sqrt(z*z - y * INV_ALPHA) - z)) * sign(x);
}

float remap_unorm_to_gaussian(float x, float truncation) {
	return inv_error_function(x * 2.0 - 1.0, truncation);
}

float pixel_cone_spread_angle(float texture_height) {
    return atan(2.0 * frame_constants.view_constants.clip_to_view._11 / texture_height);
}

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    float4 radiance_sample_count_packed = 0.0;
    uint rng = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);

    static const uint sample_count = 1;
    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        float px_off0 = 0.5;
        float px_off1 = 0.5;

        if (USE_PIXEL_FILTER) {
            const float psf_scale = 0.4;
            px_off0 += psf_scale * remap_unorm_to_gaussian(uint_to_u01_float(hash1_mut(rng)), 1e-8);
            px_off1 += psf_scale * remap_unorm_to_gaussian(uint_to_u01_float(hash1_mut(rng)), 1e-8);
        }

        const float2 pixel_center = px + float2(px_off0, px_off1);
        const float2 uv = pixel_center / DispatchRaysDimensions().xy;

        RayDesc outgoing_ray;
        {
            const ViewRayContext view_ray_context = ViewRayContext::from_uv(uv);
            const float3 ray_dir_ws = view_ray_context.ray_dir_ws();

            outgoing_ray = new_ray(
                view_ray_context.ray_origin_ws(), 
                normalize(ray_dir_ws.xyz),
                0.0,
                FLT_MAX
            );
        }

        float3 throughput = 1.0.xxx;
        float3 total_radiance = 0.0.xxx;

        float roughness_bias = 0.0;

        // TODO
        float cone_spread_angle = 0;//pixel_cone_spread_angle(DispatchRaysDimensions().y);

        [loop]
        for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
            /*if (path_length == 1 && outgoing_ray.Direction.x > -0.8) {
                throughput = 0;
            } else {
                throughput *= 2;
            }*/

            if (path_length == 1) {
                outgoing_ray.TMax = MAX_RAY_LENGTH;
            }

            const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray, cone_spread_angle);
            if (primary_hit.is_hit) {
                const float3 to_light_norm = SUN_DIRECTION;
                
                const bool is_shadowed =
                    (INDIRECT_ONLY && path_length == 0) ||
                    path_length+1 >= MAX_PATH_LENGTH ||
                    rt_is_shadowed(
                        acceleration_structure,
                        new_ray(
                            primary_hit.position,
                            to_light_norm,
                            1e-4,
                            FLT_MAX
                    ));

                GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();

                if (dot(gbuffer.normal, outgoing_ray.Direction) >= 0.0) {
                    break;
                }

                if (FURNACE_TEST && !FURNACE_TEST_EXCLUDE_DIFFUSE) {
                    gbuffer.albedo = 1.0;
                }

                //gbuffer.albedo = float3(0.966653, 0.802156, 0.323968); // Au from Mitsuba
                //gbuffer.albedo = 0;
                //gbuffer.metalness = 1.0;
                //gbuffer.roughness = 0.5;//lerp(gbuffer.roughness, 1.0, 0.8);

                if (INDIRECT_ONLY && path_length == 0) {
                    gbuffer.albedo = 1.0;
                    gbuffer.metalness = 0.0;
                }

                if (ONLY_SPECULAR_FIRST_BOUNCE && path_length == 0) {
                    gbuffer.albedo = 1.0;
                    gbuffer.metalness = 1.0;
                    //gbuffer.roughness = 0.01;
                }

                if (GREY_ALBEDO_FIRST_BOUNCE && path_length == 0) {
                    gbuffer.albedo = 0.5;
                }
                
                //gbuffer.roughness = lerp(gbuffer.roughness, 0.0, 0.8);
                //gbuffer.metalness = 1.0;
                //gbuffer.albedo = max(gbuffer.albedo, 1e-3);
                //gbuffer.roughness = 0.07;
                //gbuffer.roughness = clamp((int(primary_hit.position.x * 0.2) % 5) / 5.0, 1e-4, 1.0);

                const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, shading_basis);

                float3 wo = mul(-outgoing_ray.Direction, shading_basis);

                // Hack for shading normals facing away from the outgoing ray's direction:
                // We flip the outgoing ray along the shading normal, so that the reflection's curvature
                // continues, albeit at a lower rate.
                if (wo.z < 0.0) {
                    wo.z *= -0.25;
                    wo = normalize(wo);
                }

                LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

                if (FIREFLY_SUPPRESSION) {
                    brdf.specular_brdf.roughness = lerp(brdf.specular_brdf.roughness, 1.0, roughness_bias);
                }

                if (FURNACE_TEST && FURNACE_TEST_EXCLUDE_DIFFUSE) {
                    brdf.diffuse_brdf.albedo = 0.0.xxx;
                }

                if (!FURNACE_TEST && !(ONLY_SPECULAR_FIRST_BOUNCE && path_length == 0)) {
                    const float3 brdf_value = brdf.evaluate_directional_light(wo, wi);
                    const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                    total_radiance += throughput * brdf_value * light_radiance * max(0.0, wi.z);

                    #if 0
                        const float3 pos_ws = primary_hit.position;
                        float4 pos_vs = mul(frame_constants.view_constants.world_to_view, float4(pos_ws, 1));
                        const float view_dot = -normalize(pos_vs.xyz).z;

                        float3 v_ws = normalize(mul(frame_constants.view_constants.view_to_world, float4(0, 0, -1, 0)).xyz);

                        total_radiance +=
                            throughput *
                            100 * smoothstep(0.997, 1.0, view_dot) * gbuffer.albedo * max(0.0, dot(gbuffer.normal, -v_ws)) / M_PI;
                    #endif
                }

                float3 urand;

                #if 0
                if (path_length == 0) {
                    const uint noise_offset = frame_constants.frame_index;

                    urand = bindless_textures[1][
                        (px + int2(noise_offset * 59, noise_offset * 37)) & 255
                    ].xyz * 255.0 / 256.0 + 0.5 / 256.0;

                    urand.x += uint_to_u01_float(hash1(frame_constants.frame_index));
                    urand.y += uint_to_u01_float(hash1(frame_constants.frame_index + 103770841));
                    urand.z += uint_to_u01_float(hash1(frame_constants.frame_index + 828315679));

                    urand = frac(urand);
                } else
                #endif
                {
                    urand = float3(
                        uint_to_u01_float(hash1_mut(rng)),
                        uint_to_u01_float(hash1_mut(rng)),
                        uint_to_u01_float(hash1_mut(rng)));
                }

                BrdfSample brdf_sample = brdf.sample(wo, urand);

                if (brdf_sample.is_valid()) {
                    if (FIREFLY_SUPPRESSION) {
                        roughness_bias = lerp(roughness_bias, 1.0, 0.5 * brdf_sample.approx_roughness);
                    }

                    outgoing_ray.Origin = primary_hit.position;
                    outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
                    outgoing_ray.TMin = 1e-4;
                    throughput *= brdf_sample.value_over_pdf;
                } else {
                    break;
                }

                if (FURNACE_TEST) {
                    total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
                    break;
                }

                // Russian roulette
                if (path_length >= RUSSIAN_ROULETTE_START_PATH_LENGTH) {
                    const float rr_coin = uint_to_u01_float(hash1_mut(rng));
                    const float continue_p = max(gbuffer.albedo.r, max(gbuffer.albedo.g, gbuffer.albedo.b));
                    if (rr_coin > continue_p) {
                        break;
                    } else {
                        throughput /= continue_p;
                    }
                }
            } else {
                total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
                break;
            }
        }

        radiance_sample_count_packed += float4(total_radiance, 1.0);
    }

    float4 cur = radiance_sample_count_packed;
    float4 prev = output_tex[px];

    //if (prev.w < 32)
    {
        float tsc = cur.w + prev.w;
        float lrp = cur.w / max(1.0, tsc);
        cur.rgb /= max(1.0, cur.w);

        output_tex[px] = float4(lerp(prev.rgb, cur.rgb, lrp), max(1, tsc));
    }
}
