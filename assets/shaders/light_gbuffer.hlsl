#include "inc/samplers.hlsl"
#include "inc/frame_constants.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/tonemap.hlsl"
#include "inc/rt.hlsl"
#include "inc/brdf.hlsl"
#include "inc/brdf_lut.hlsl"
#include "inc/layered_brdf.hlsl"
#include "inc/uv.hlsl"
#include "inc/bindless_textures.hlsl"
#include "rtr/rtr_settings.hlsl"
#include "wrc/wrc_settings.hlsl"
#include "surfel_gi/bindings.hlsl"
#include "wrc/bindings.hlsl"
#include "rtdgi/near_field_settings.hlsl"

#include "inc/hash.hlsl"
#include "inc/color.hlsl"

#define USE_RTR 1
#define USE_RTDGI 1

// Loses rim lighting on rough surfaces, but can be cleaner, especially without reflection restir
#define USE_DIFFUSE_GI_FOR_ROUGH_SPEC 0
#define USE_DIFFUSE_GI_FOR_ROUGH_SPEC_MIN_ROUGHNESS 0.7

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float> shadow_mask_tex;
[[vk::binding(3)]] Texture2D<float4> ssgi_tex;  // TODO: nuke
[[vk::binding(4)]] Texture2D<float4> rtr_tex;
[[vk::binding(5)]] Texture2D<float4> rtdgi_tex;
DEFINE_SURFEL_GI_BINDINGS(6, 7, 8, 9, 10, 11, 12)
DEFINE_WRC_BINDINGS(13)
[[vk::binding(14)]] RWTexture2D<float4> temporal_output_tex;
[[vk::binding(15)]] RWTexture2D<float4> output_tex;
[[vk::binding(16)]] TextureCube<float4> unconvolved_sky_cube_tex;
[[vk::binding(17)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(18)]] cbuffer _ {
    float4 output_tex_size;
    uint debug_shading_mode;
    uint debug_show_wrc;
};

#include "surfel_gi/lookup.hlsl"
#include "wrc/lookup.hlsl"
#include "wrc/wrc_intersect_probe_grid.hlsl"

#define SHADING_MODE_DEFAULT 0
#define SHADING_MODE_NO_TEXTURES 1
#define SHADING_MODE_DIFFUSE_GI 2
#define SHADING_MODE_REFLECTIONS 3
#define SHADING_MODE_RTX_OFF 4
#define SHADING_MODE_SURFEL_GI 5

#include "inc/atmosphere.hlsl"
#include "inc/sun.hlsl"

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);
    uint rng = hash3(uint3(px, frame_constants.frame_index));

    RayDesc outgoing_ray;
    const ViewRayContext view_ray_context = ViewRayContext::from_uv(uv);
    {
        outgoing_ray = new_ray(
            view_ray_context.ray_origin_ws(), 
            view_ray_context.ray_dir_ws(),
            0.0,
            FLT_MAX
        );
    }

    const float depth = depth_tex[px];

    if (debug_show_wrc) {
        float4 hit_color = wrc_intersect_probe_grid(
            outgoing_ray.Origin,
            outgoing_ray.Direction,
            -depth_to_view_z(depth) * length(view_ray_context.ray_dir_vs_h.xyz)
        );

        if (hit_color.a != 0) {
            output_tex[px] = hit_color;
            return;
        }
    }

    if (depth == 0.0) {
        // Render the sun disk

        // Allow the size to be changed, but don't go below the real sun's size,
        // so that we have something in the sky.
        const float real_sun_angular_radius = 0.53 * 0.5 * PI / 180.0;
        const float sun_angular_radius_cos = min(cos(real_sun_angular_radius), frame_constants.sun_angular_radius_cos);

        // Conserve the sun's energy by making it dimmer as it increases in size
        // Note that specular isn't quite correct with this since we're not using area lights.
        float current_sun_angular_radius = acos(sun_angular_radius_cos);
        float sun_radius_ratio = real_sun_angular_radius / current_sun_angular_radius;

        float3 output = unconvolved_sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
        if (dot(outgoing_ray.Direction, SUN_DIRECTION) > sun_angular_radius_cos) {
            // TODO: what's the correct value?
            output += 800 * sun_color_in_direction(outgoing_ray.Direction) * sun_radius_ratio * sun_radius_ratio;
        }
        
        temporal_output_tex[px] = float4(output, 1);
        output_tex[px] = float4(output, 1);
        return;
    }

    float4 pt_cs = float4(uv_to_cs(uv), depth, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

    const float3 to_light_norm = SUN_DIRECTION;
    /*const float3 to_light_norm = sample_sun_direction(
        float2(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))),
        true
    );*/

    float shadow_mask = shadow_mask_tex[px].x;

    if (debug_shading_mode == SHADING_MODE_RTX_OFF) {
        shadow_mask = 1;
    }

    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[px])).unpack();

    if (debug_shading_mode == SHADING_MODE_NO_TEXTURES) {
        gbuffer.albedo = 0.5;
    }

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
    const float3 brdf_value = brdf.evaluate_directional_light(wo, wi) * max(0.0, wi.z);
    const float3 light_radiance = shadow_mask * SUN_COLOR;
    float3 total_radiance = brdf_value * light_radiance;

    total_radiance += gbuffer.emissive;

    float3 gi_irradiance = 0.0.xxx;

    if (debug_shading_mode != SHADING_MODE_RTX_OFF) {
        gi_irradiance = rtdgi_tex[px].rgb;
    }

    total_radiance += gi_irradiance
        * brdf.diffuse_brdf.albedo
        #if !LAYERED_BRDF_FORCE_DIFFUSE_ONLY
            * brdf.energy_preservation.preintegrated_transmission_fraction
        #endif
        ;

    if (USE_RTR && debug_shading_mode != SHADING_MODE_RTX_OFF) {
        float3 rtr_radiance;

        #if !RTR_RENDER_SCALED_BY_FG
            rtr_radiance = rtr_tex[px].xyz * brdf.energy_preservation.preintegrated_reflection;
        #else
            rtr_radiance = rtr_tex[px].xyz;
        #endif

        if (USE_DIFFUSE_GI_FOR_ROUGH_SPEC) {
            rtr_radiance = lerp(
                rtr_radiance,
                gi_irradiance * brdf.energy_preservation.preintegrated_reflection,
                smoothstep(USE_DIFFUSE_GI_FOR_ROUGH_SPEC_MIN_ROUGHNESS, 1.0, gbuffer.roughness));
        }

        if (debug_shading_mode == SHADING_MODE_NO_TEXTURES) {
            GbufferData true_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[px])).unpack();
            LayeredBrdf true_brdf = LayeredBrdf::from_gbuffer_ndotv(true_gbuffer, wo.z);
            rtr_radiance /= true_brdf.energy_preservation.preintegrated_reflection;
        }
        
        total_radiance += rtr_radiance;
    }

    temporal_output_tex[px] = float4(total_radiance, 1.0);

    float3 output = total_radiance;

    if (debug_shading_mode == SHADING_MODE_REFLECTIONS) {
        #if !RTR_RENDER_SCALED_BY_FG
            output = rtr_tex[px].xyz * brdf.energy_preservation.preintegrated_reflection;
        #else
            output = rtr_tex[px].xyz;
        #endif

        if (USE_DIFFUSE_GI_FOR_ROUGH_SPEC) {
            output = lerp(
                output,
                gi_irradiance * brdf.energy_preservation.preintegrated_reflection,
                smoothstep(USE_DIFFUSE_GI_FOR_ROUGH_SPEC_MIN_ROUGHNESS, 1.0, gbuffer.roughness));
        }

        GbufferData true_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[px])).unpack();
        LayeredBrdf true_brdf = LayeredBrdf::from_gbuffer_ndotv(true_gbuffer, wo.z);
        output /= true_brdf.energy_preservation.preintegrated_reflection;
    }

    if (debug_shading_mode == SHADING_MODE_DIFFUSE_GI) {
        output = gi_irradiance;
    }

    if (debug_shading_mode == SHADING_MODE_SURFEL_GI) {
        output = lookup_surfel_gi(pt_ws.xyz, gbuffer.normal);
    }

    //output = gbuffer.albedo;
    output_tex[px] = float4(output, 1.0);
}
