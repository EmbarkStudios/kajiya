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

#include "inc/hash.hlsl"
#include "inc/color.hlsl"

#include "csgi/common.hlsl"

#define USE_SSGI 0
#define USE_CSGI 1
#define USE_RTR 1
#define USE_RTDGI 1

#define SSGI_INTENSITY_BIAS 0.0

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float> shadow_mask_tex;
[[vk::binding(3)]] Texture2D<float4> ssgi_tex;
[[vk::binding(4)]] Texture2D<float4> rtr_tex;
[[vk::binding(5)]] Texture2D<float4> rtdgi_tex;
[[vk::binding(6)]] RWTexture2D<float4> temporal_output_tex;
[[vk::binding(7)]] RWTexture2D<float4> output_tex;
[[vk::binding(8)]] Texture3D<float4> csgi_indirect_tex[CSGI_CASCADE_COUNT];
[[vk::binding(9)]] TextureCube<float4> unconvolved_sky_cube_tex;
[[vk::binding(10)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(11)]] cbuffer _ {
    float4 output_tex_size;
    uint debug_shading_mode;
};

#define SHADING_MODE_DEFAULT 0
#define SHADING_MODE_NO_TEXTURES 1
#define SHADING_MODE_DIFFUSE_GI 2
#define SHADING_MODE_REFLECTIONS 3
#define SHADING_MODE_RTX_OFF 4

#include "csgi/lookup.hlsl"

#include "inc/atmosphere.hlsl"
#include "inc/sun.hlsl"

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

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

    const float depth = depth_tex[px];
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

    float3 csgi_irradiance = 0;

    if (USE_CSGI && debug_shading_mode != SHADING_MODE_RTX_OFF) {
        if (USE_RTDGI) {
            gi_irradiance = rtdgi_tex[px].rgb;
        } else {
            float3 to_eye = get_eye_position() - pt_ws.xyz;
            float3 pseudo_bent_normal = normalize(normalize(to_eye) + gbuffer.normal);
            
            csgi_irradiance = lookup_csgi(
                pt_ws.xyz,
                gbuffer.normal,
                CsgiLookupParams::make_default()
                    .with_bent_normal(pseudo_bent_normal)
            );
            gi_irradiance = csgi_irradiance;
        }
    }

    const float4 ssgi = ssgi_tex[px];
    #if USE_SSGI
        // HACK: need directionality in GI so that it can be properly masked.
        // If simply masking with the AO term, it tends to over-darken.
        // Reduce some of the occlusion, but for energy conservation, also reduce
        // the light added.
        const float4 biased_ssgi = lerp(ssgi, float4(0, 0, 0, 1), SSGI_INTENSITY_BIAS);
        gi_irradiance *= biased_ssgi.a;
        gi_irradiance += biased_ssgi.rgb;
    #endif

    total_radiance += gi_irradiance
        * brdf.diffuse_brdf.albedo
        * brdf.energy_preservation.preintegrated_transmission_fraction
        ;

    if (USE_RTR && debug_shading_mode != SHADING_MODE_RTX_OFF) {
        float3 rtr_radiance;

        #if !RTR_RENDER_SCALED_BY_FG
            rtr_radiance = rtr_tex[px].xyz * brdf.energy_preservation.preintegrated_reflection;
        #else
            rtr_radiance = rtr_tex[px].xyz;
        #endif

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

        GbufferData true_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[px])).unpack();
        LayeredBrdf true_brdf = LayeredBrdf::from_gbuffer_ndotv(true_gbuffer, wo.z);
        output /= true_brdf.energy_preservation.preintegrated_reflection;
    }

    if (debug_shading_mode == SHADING_MODE_DIFFUSE_GI) {
        output = gi_irradiance;
    }

    //output = gbuffer.albedo;
    output_tex[px] = float4(output, 1.0);
}
