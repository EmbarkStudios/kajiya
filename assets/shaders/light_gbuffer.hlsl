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

#define USE_SSGI 0
#define USE_RTR 0
#define USE_RTDGI 1

#define SSGI_INTENSITY_BIAS 0.0

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float> shadow_mask_tex;
[[vk::binding(3)]] Texture2D<float4> ssgi_tex;
[[vk::binding(4)]] Texture2D<float4> rtr_tex;
[[vk::binding(5)]] Texture2D<float4> rtdgi_tex;
DEFINE_SURFEL_GI_BINDINGS(6, 7, 8, 9, 10, 11)
DEFINE_WRC_BINDINGS(12)
[[vk::binding(13)]] RWTexture2D<float4> temporal_output_tex;
[[vk::binding(14)]] RWTexture2D<float4> output_tex;
[[vk::binding(15)]] TextureCube<float4> unconvolved_sky_cube_tex;
[[vk::binding(16)]] TextureCube<float4> sky_cube_tex;
[[vk::binding(17)]] cbuffer _ {
    float4 output_tex_size;
    uint debug_shading_mode;
    uint debug_show_wrc;
};

#include "surfel_gi/lookup.hlsl"
#include "wrc/lookup.hlsl"
#include "wrc/wrc_intersect_probe_grid.hlsl"

#define SHADING_MODE_DEFAULT 2
#define SHADING_MODE_NO_TEXTURES 1
#define SHADING_MODE_DIFFUSE_GI 0
#define SHADING_MODE_REFLECTIONS 3
#define SHADING_MODE_RTX_OFF 4
#define SHADING_MODE_SURFEL_GI 5

#include "inc/atmosphere.hlsl"
#include "inc/sun.hlsl"

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

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
        #if 1
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
        #else
            float3 output = atmosphere_default(outgoing_ray.Direction, SUN_DIRECTION);
            //float3 output = outgoing_ray.Direction * 0.5 + 0.5;
        #endif
        
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
    //gbuffer.roughness = 0.9;
    //gbuffer.metalness = 0;
    //gbuffer.metalness = 1;

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

    //const float3 radiance = (spec.value + spec.transmission_fraction * diff.value) * max(0.0, wi.z);
    //const float3 light_radiance = shadow_mask * SUN_COLOR;
    //total_radiance += radiance * light_radiance;

    //res.xyz += radiance * SUN_COLOR + ambient;
    //res.xyz += albedo;
    //res.xyz += metalness;
    //res.xyz = metalness;
    //res.xyz = 0.0001 / depth;
    //res.xyz = frac(pt_ws.xyz * 10.0);
    //res.xyz = brdf_d * 0.1;

    //res.xyz = 1.0 - exp(-res.xyz);
    //total_radiance = preintegrated_specular_brdf_fg(specular_brdf.albedo, specular_brdf.roughness, wo.z);

    float3 gi_irradiance = 0.0.xxx;

    if (debug_shading_mode != SHADING_MODE_RTX_OFF) {
        gi_irradiance = rtdgi_tex[px].rgb;
    }

    const float4 ssgi = ssgi_tex[px].gbar;
    #if USE_SSGI
        // HACK: need directionality in GI so that it can be properly masked.
        // If simply masking with the AO term, it tends to over-darken.
        // Reduce some of the occlusion, but for energy conservation, also reduce
        // the light added.
        const float4 biased_ssgi = lerp(ssgi, float4(0, 0, 0, 1), SSGI_INTENSITY_BIAS);
        //gi_irradiance *= biased_ssgi.a;
        //gi_irradiance += biased_ssgi.rgb;
        #if USE_SSGI_NEAR_FIELD
            gi_irradiance += biased_ssgi.rgb;
        #endif
    #endif

    //gi_irradiance = ssgi.a;

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

        if (debug_shading_mode == SHADING_MODE_NO_TEXTURES) {
            GbufferData true_gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_tex[px])).unpack();
            LayeredBrdf true_brdf = LayeredBrdf::from_gbuffer_ndotv(true_gbuffer, wo.z);
            rtr_radiance /= true_brdf.energy_preservation.preintegrated_reflection;
        }
        
        total_radiance += rtr_radiance;
    }

    //total_radiance = gbuffer.albedo * (ssgi.a + ssgi.rgb);

    temporal_output_tex[px] = float4(total_radiance, 1.0);

    float3 output = total_radiance;
    
    #if 0
        float4 pos_vs = mul(frame_constants.view_constants.world_to_view, pt_ws);
        const float view_dot = -normalize(pos_vs.xyz).z;

        float3 v_ws = normalize(mul(frame_constants.view_constants.view_to_world, float4(0, 0, -1, 0)).xyz);

        output +=
            smoothstep(0.997, 1.0, view_dot) * gbuffer.albedo * max(0.0, dot(gbuffer.normal, -v_ws)) / M_PI;
    #endif

    //float ex = calculate_luma(rtr_tex[px].xyz);
    //float ex2 = rtr_tex[px].w;
    //output = 1e-1 * abs(ex * ex - ex2) / max(1e-8, ex);

    //output = rtr_tex[px].www / 16.0;

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

    //const float3 bent_normal_dir = mul(frame_constants.view_constants.view_to_world, float4(ssgi.xyz, 0)).xyz;
    //output = pow((bent_normal_dir) * 0.5 + 0.5, 2);
    //output = bent_normal_dir * 0.5 + 0.5;
    //output = pow(gbuffer.normal.xyz * 0.5 + 0.5, 2);

    if (debug_shading_mode == SHADING_MODE_DIFFUSE_GI) {
        output = gi_irradiance;
    }

    if (debug_shading_mode == SHADING_MODE_SURFEL_GI) {
        output = lookup_surfel_gi(pt_ws.xyz, gbuffer.normal);
    }

    //output = gbuffer.emissive;

    // Hacky visual test of volumetric scattering
    /*if (frame_constants.global_fog_thickness > 0.0) {
        float3 scattering = 0.0.xxx;
        float prev_t = 0.0;
        float sigma_s = 0.07 * frame_constants.global_fog_thickness;
        float sigma_e = sigma_s * 0.4;

        float3 eye_to_pt = pt_ws.xyz - get_eye_position();
        const float total_ray_length = length(eye_to_pt);
        const int k_samples = 6;//clamp(int(0.25 * total_ray_length), 3, 8);

        //uint rng = hash3(uint3(px * 2, 0*frame_constants.frame_index));
        //float t_offset = uint_to_u01_float(hash1_mut(rng));
        float t_offset;
        {
            const uint noise_offset = frame_constants.frame_index;
            t_offset = bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][
                (px + int2(noise_offset * 59, noise_offset * 37)) & 255
            ].x * 255.0 / 256.0 + 0.5 / 256.0;
        }

        float transmittance = 1.0;

        for (int k = 0; k < k_samples; ++k) {
            const float t = float(k + t_offset) / float(k_samples);
            const float step_size = (t - prev_t);

            const float3 air_ws = get_eye_position() + eye_to_pt * lerp(prev_t, t, 0.5);
            const float3 gi_color = lookup_gi(
                air_ws,
                0.0.xxx,
                GiLookupParams::make_default()
                    .with_sample_phase(0.6, outgoing_ray.Direction)
            );

            const float local_density = 1;//max(1e-5, lerp(0.1, 1.0, smoothstep(0.0, 1.0, air_ws.x)));
            
            // Based on https://www.shadertoy.com/view/XlBSRz
            // (See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/)
            float3 scattered_light = gi_color * (sigma_s * local_density);
            float3 s_int = scattered_light * (1.0 - exp(-total_ray_length * step_size * sigma_e * local_density)) / (sigma_e * local_density);
            scattering += transmittance * s_int;
            transmittance *= exp(-total_ray_length * step_size * sigma_e * local_density);

            prev_t = t;
        }

        output *= transmittance;
        output += scattering;
        //output = scattering;
    }*/

    //output = gbuffer.metalness;
    //output = gbuffer.roughness;
    //output = gbuffer.albedo;
    //output = gbuffer.normal * 0.5 + 0.5;
    //output = ssgi.rgb;

    //output = shadow_mask;
    //output = shadow_mask_tex[px].x;
    //output = shadow_mask_tex[px].y * 10;
    //output = sqrt(shadow_mask_tex[px].y) * 0.1;
    //output = shadow_mask_tex[px].z * 0.1;
    //output = rtr_tex[px].rgb;

    //output.xz += 1;
    //output.rgb /= max(1e-5, calculate_luma(output));

    output_tex[px] = float4(output, 1.0);
}
