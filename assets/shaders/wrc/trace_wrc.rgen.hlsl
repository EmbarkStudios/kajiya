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
#include "../inc/mesh.hlsl"
#include "../inc/sh.hlsl"
#include "../inc/quasi_random.hlsl"
#include "../surfel_gi/bindings.hlsl"
#include "wrc_settings.hlsl"


[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] TextureCube<float4> sky_cube_tex;
DEFINE_SURFEL_GI_BINDINGS(1, 2, 3, 4, 5, 6)
[[vk::binding(7)]] RWTexture2D<float4> radiance_atlas_out_tex;

#include "../surfel_gi/lookup.hlsl"
#include "../inc/sun.hlsl"

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool USE_EMISSIVE = true;
static const bool USE_SURFEL_GI = true;
static const bool USE_BLEND_OUTPUT = true;

float3 sample_environment_light(float3 dir) {
    //return 0.0.xxx;
    return sky_cube_tex.SampleLevel(sampler_llr, dir, 0).rgb;
    /*return atmosphere_default(dir, SUN_DIRECTION);

    float3 col = (dir.zyx * float3(1, 1, -1) * 0.5 + float3(0.6, 0.5, 0.5)) * 0.75;
    col = lerp(col, 1.3.xxx * calculate_luma(col), smoothstep(-0.2, 1.0, dir.y).xxx);
    return col;*/
}

float pack_dist(float x) {
    return x;
}

float unpack_dist(float x) {
    return x;
}

[shader("raygeneration")]
void main() {
    const uint probe_idx = DispatchRaysIndex().x / (WRC_PROBE_DIMS * WRC_PROBE_DIMS);
    const uint probe_px_idx = DispatchRaysIndex().x % (WRC_PROBE_DIMS * WRC_PROBE_DIMS);
    const uint2 probe_px = uint2(probe_px_idx % WRC_PROBE_DIMS, probe_px_idx / WRC_PROBE_DIMS);
    const uint2 tile = wrc_probe_idx_to_atlas_tile(probe_idx);
    const uint2 atlas_px = tile * WRC_PROBE_DIMS + probe_px;
    const uint3 probe_coord = wrc_probe_idx_to_coord(probe_idx);

    float3 irradiance_sum = 0;
    float valid_sample_count = 0;
    float hit_count = 0;
    float2 hit_dist_wt = 0;
    const uint sample_count = 1;

    for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        valid_sample_count += 1.0;

        RayDesc outgoing_ray;
        {
            float3 dir;
            if (USE_BLEND_OUTPUT) {
                dir = octa_decode((probe_px + r2_sequence(frame_constants.frame_index)) / WRC_PROBE_DIMS);
            } else {
                dir = octa_decode((probe_px + 0.5) / WRC_PROBE_DIMS);
            }

            outgoing_ray = new_ray(
                wrc_probe_center(probe_coord),
                dir,
                WRC_MIN_TRACE_DIST,
                FLT_MAX
            );
        }

        // ----

        float roughness_bias = 0.0;

        {
            const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
                .with_cone(RayCone::from_spread_angle(0.1))
                .with_cull_back_faces(true)
                .with_path_length(1)  // +1 because this is indirect light
                .trace(acceleration_structure);

            if (primary_hit.is_hit) {
                const float3 to_light_norm = SUN_DIRECTION;
                
                const bool is_shadowed = rt_is_shadowed(
                    acceleration_structure,
                    new_ray(
                        primary_hit.position,
                        to_light_norm,
                        1e-4,
                        FLT_MAX
                ));

                hit_dist_wt += float2(pack_dist(primary_hit.ray_t), 1);

                GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();

                const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, tangent_to_world);

                float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);

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

                const float3 brdf_value = brdf.evaluate_directional_light(wo, wi);
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                irradiance_sum += brdf_value * light_radiance * max(0.0, wi.z);

                if (USE_EMISSIVE) {
                    irradiance_sum += gbuffer.emissive;
                }

                if (USE_SURFEL_GI) {
                    irradiance_sum += lookup_surfel_gi(primary_hit.position, gbuffer.normal) * gbuffer.albedo;
                }
                
                hit_count += 1.0;
            } else {
                hit_dist_wt += float2(pack_dist(1e5), 1);
                irradiance_sum += sample_environment_light(outgoing_ray.Direction);
            }
        }
    }

    irradiance_sum /= max(1.0, valid_sample_count);

    float avg_dist = unpack_dist(hit_dist_wt.x / max(1, hit_dist_wt.y));

    //radiance_atlas_out_tex[atlas_px] = float4(float2(probe_px + 0.5) / WRC_PROBE_DIMS, 0.0.xx);

    if (USE_BLEND_OUTPUT) {
        radiance_atlas_out_tex[atlas_px] = lerp(
            radiance_atlas_out_tex[atlas_px],
            float4(irradiance_sum, avg_dist),
            1.0 / 8.0);
    } else {
        radiance_atlas_out_tex[atlas_px] = float4(irradiance_sum, avg_dist);
    }
}
