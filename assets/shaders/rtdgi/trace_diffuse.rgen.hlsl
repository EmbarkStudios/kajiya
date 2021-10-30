#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/blue_noise.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/lights/triangle.hlsl"
#include "../inc/reservoir.hlsl"
#include "../surfel_gi/bindings.hlsl"
#include "restir_settings.hlsl"

// Should be 1, but rarely matters for the diffuse bounce, so might as well save a few cycles.
#define USE_SOFT_SHADOWS 0

#define USE_SURFEL_GI 1

#define USE_TEMPORAL_JITTER 1
#define USE_SHORT_RAYS_ONLY 0
#define SHORT_RAY_SIZE_VOXEL_CELLS 4.0
#define ROUGHNESS_BIAS 0.5
#define SUPPRESS_GI_FOR_NEAR_HITS 1
#define USE_SCREEN_GI_REPROJECTION 0
#define USE_SWIZZLE_TILE_PIXELS 0

#define USE_EMISSIVE 1
#define USE_LIGHTS 1

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float3> half_view_normal_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float4> reprojected_gi_tex;
[[vk::binding(3)]] Texture2D<float> ssao_tex;
DEFINE_BLUE_NOISE_SAMPLER_BINDINGS(4, 5, 6)
[[vk::binding(7)]] Texture2D<float4> reprojection_tex;
DEFINE_SURFEL_GI_BINDINGS(8, 9, 10, 11, 12, 13)
[[vk::binding(14)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(15)]] RWTexture2D<float3> candidate_irradiance_out_tex;
[[vk::binding(16)]] RWTexture2D<float4> candidate_hit_out_tex;
[[vk::binding(17)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

#include "../surfel_gi/lookup.hlsl"
#include "candidate_ray_dir.hlsl"

static const float SKY_DIST = 1e4;

uint2 reservoir_payload_to_px(uint payload) {
    return uint2(payload & 0xffff, payload >> 16);
}

struct TraceResult {
    float3 out_value;
    float3 hit_normal_ws;
    float hit_t;
};

TraceResult do_the_thing(uint2 px, inout uint rng, RayDesc outgoing_ray, float3 primary_hit_normal) {
    float3 total_radiance = 0.0.xxx;
    float3 hit_normal_ws = -outgoing_ray.Direction;

    #if USE_SHORT_RAYS_ONLY
        outgoing_ray.TMax = TODO;
    #else
        outgoing_ray.TMax = SKY_DIST;
    #endif

    float hit_t = outgoing_ray.TMax;

    // TODO: cone spread angle
    const GbufferPathVertex primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
        .with_cone_width(0.05)
        .with_cull_back_faces(true)
        .with_path_length(1)
        .trace(acceleration_structure);

    if (primary_hit.is_hit) {
        hit_t = primary_hit.ray_t;
        GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
        hit_normal_ws = gbuffer.normal;

        // Project the sample into clip space, and check if it's on-screen
        /*const float3 primary_hit_cs = position_world_to_clip(primary_hit.position);
        const float2 primary_hit_uv = cs_to_uv(primary_hit_cs.xy);
        const float primary_hit_screen_depth = depth_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0);
        const GbufferDataPacked primary_hit_screen_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0)));
        const float3 primary_hit_screen_normal_ws = primary_hit_screen_gbuffer.unpack_normal();
        bool is_on_screen =
            all(abs(primary_hit_cs.xy) < 1.0)
            && inverse_depth_relative_diff(primary_hit_cs.z, primary_hit_screen_depth) < 5e-3
            && dot(primary_hit_screen_normal_ws, -outgoing_ray.Direction) > 0.0
            && dot(primary_hit_screen_normal_ws, gbuffer.normal) > 0.7
            ;*/

        // If it is on-screen, we'll try to use its reprojected radiance from the previous frame
        /*float4 reprojected_radiance = 0;
        if (is_on_screen) {
            reprojected_radiance =
                reprojected_gi_tex.SampleLevel(sampler_nnc, primary_hit_uv, 0);

            // Check if the temporal reprojection is valid.
            is_on_screen = reprojected_radiance.w > 0;
        }*/

        gbuffer.roughness = lerp(gbuffer.roughness, 1.0, ROUGHNESS_BIAS);
        const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
        const float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);
        const LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

        // Sun
        float3 sun_radiance = SUN_COLOR;
        if (any(sun_radiance) > 0) {
            const float3 to_light_norm = sample_sun_direction(
                blue_noise_for_pixel(px, frame_constants.frame_index).xy,
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
            const float3 light_radiance = is_shadowed ? 0.0 : sun_radiance;
            total_radiance += brdf_value * light_radiance;
        }

        if (USE_EMISSIVE) {
            total_radiance += gbuffer.emissive;
        }

        /*if (USE_SCREEN_GI_REPROJECTION && is_on_screen) {
            total_radiance += reprojected_radiance.rgb * gbuffer.albedo;
        } else */{
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
                            !is_shadowed ? (triangle_light.radiance() * brdf_value / light_sample.pdf.value) : 0;
                    }
                }
            }

            if (USE_SURFEL_GI) {
                float3 gi = lookup_surfel_gi(
                    primary_hit.position,
                    gbuffer.normal
                );

                total_radiance += gi * gbuffer.albedo;
            }
        }
    } else {
        #if USE_SHORT_RAYS_ONLY
            /*const float3 csgi_lookup_pos = outgoing_ray.Origin + outgoing_ray.Direction * max(0.0, outgoing_ray.TMax - csgi_blended_voxel_size(origin_cascade_idx).x);

            total_radiance += lookup_csgi(
                csgi_lookup_pos,
                0.0.xxx,    // don't offset by any normal
                CsgiLookupParams::make_default()
	                .with_sample_directional_radiance(outgoing_ray.Direction)
                    //.with_directional_radiance_phong_exponent(8)
            );*/
        #else
            total_radiance += sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
        #endif
    }

    float3 out_value = total_radiance;

    //out_value /= reservoir.p_sel;

    TraceResult result;
    result.out_value = out_value;
    result.hit_t = hit_t;
    result.hit_normal_ws = hit_normal_ws;
    return result;
}

[shader("raygeneration")]
void main() {
    uint2 px;
    if (USE_SWIZZLE_TILE_PIXELS) {
        const uint2 orig_px = DispatchRaysIndex().xy;

        // TODO: handle screen edge
        px = (orig_px & 7) * 8 + ((orig_px / 8) & 7) + (orig_px & ~63u);
    } else {
        px = DispatchRaysIndex().xy;
    }

    const uint2 hi_px_subpixels[4] = {
        uint2(0, 0),
        uint2(1, 1),
        uint2(1, 0),
        uint2(0, 1),
    };

    const int2 hi_px_offset = hi_px_subpixels[frame_constants.frame_index & 3];
    const uint2 hi_px = px * 2 + hi_px_offset;
    
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    const float3 normal_vs = half_view_normal_tex[px];
    const float3 normal_ws = direction_view_to_world(normal_vs);

    const float3x3 tangent_to_world = build_orthonormal_basis(normal_ws);

    const float3 outgoing_dir = rtdgi_candidate_ray_dir(px, tangent_to_world);

    RayDesc outgoing_ray;
    outgoing_ray.Direction = outgoing_dir;
    outgoing_ray.Origin = view_ray_context.biased_secondary_ray_origin_ws();
    outgoing_ray.TMin = 0;
    outgoing_ray.TMax = SKY_DIST;

    uint rng = hash3(uint3(px, frame_constants.frame_index));
    TraceResult result = do_the_thing(px, rng, outgoing_ray, normal_ws);

    candidate_irradiance_out_tex[px] = result.out_value;
    candidate_hit_out_tex[px] = float4(result.hit_normal_ws, result.hit_t);
}
