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
#include "../inc/sun.hlsl"
#include "../inc/atmosphere.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] RWTexture2D<float4> output_tex;

static const uint MAX_PATH_LENGTH = 3;
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool FURNACE_TEST = false;
static const bool FURNACE_TEST_EXCLUDE_DIFFUSE = false;
static const bool USE_PIXEL_FILTER = false;
static const bool INDIRECT_ONLY = true;

float3 sample_environment_light(float3 dir) {
    return 0.0.xxx;

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


[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    uint rng = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);

    float px_off0 = 0.5;
    float px_off1 = 0.5;

    if (USE_PIXEL_FILTER) {
        const float psf_scale = 0.85;
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

    [loop]
    for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
        /*if (path_length == 1 && outgoing_ray.Direction.x > -0.8) {
            throughput = 0;
        } else {
            throughput *= 2;
        }*/

        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
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
            if (FURNACE_TEST && !FURNACE_TEST_EXCLUDE_DIFFUSE) {
                gbuffer.albedo = 1.0;
            }

            if (INDIRECT_ONLY && path_length == 0) {
                gbuffer.albedo = 1.0;
                gbuffer.metalness = 0.0;
            }
            //gbuffer.roughness = lerp(gbuffer.roughness, 0.0, 0.8);
            //gbuffer.metalness = 1.0;
            //gbuffer.albedo = max(gbuffer.albedo, 1e-3);
            //gbuffer.albedo = float3(1, 0.765557, 0.336057);
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

            if (!FURNACE_TEST) {
                const float3 brdf_value = brdf.evaluate(wo, wi);
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                total_radiance += throughput * brdf_value * light_radiance;
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
                roughness_bias = lerp(roughness_bias, 1.0, 0.5 * brdf_sample.approx_roughness);
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
        } else {
            total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
            break;
        }
    }

    float4 prev_radiance = output_tex[px];
    //if (prev_radiance.w < 100)
    {
        output_tex[px] = float4(total_radiance, 1.0) + prev_radiance;
    }
}
