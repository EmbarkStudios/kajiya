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

#include "inc/hash.hlsl"
#include "inc/color.hlsl"

#define USE_SSGI 0
#define USE_CSGI 1
#define USE_RTR 1
#define USE_RTDGI 1

#define SSGI_INTENSITY_BIAS 0.0

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float> sun_shadow_mask_tex;
[[vk::binding(3)]] Texture2D<float4> ssgi_tex;
[[vk::binding(4)]] Texture2D<float4> rtr_tex;
[[vk::binding(5)]] Texture2D<float4> rtdgi_tex;
[[vk::binding(6)]] RWTexture2D<float4> temporal_output_tex;
[[vk::binding(7)]] RWTexture2D<float4> output_tex;
[[vk::binding(8)]] Texture3D<float4> csgi_direct_tex;
[[vk::binding(9)]] Texture3D<float4> csgi_indirect_tex;
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

#include "csgi/common.hlsl"
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
        #if 1
            float3 output = sky_cube_tex.SampleLevel(sampler_llr, outgoing_ray.Direction, 0).rgb;
            if (dot(outgoing_ray.Direction, SUN_DIRECTION) > 0.999958816) { // cos(0.52 degrees)
                // TODO: what's the correct value?
                output += SUN_COLOR * 200;
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
    
    float shadow_mask = sun_shadow_mask_tex[px];
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

    uint rng = hash3(asuint(pt_ws.xyz * 3.0));
    rng = hash2(uint2(rng, frame_constants.frame_index));
    //total_radiance += uint_id_to_color(pt_hash);

    float3 gi_irradiance = 0.0.xxx;

    float3 csgi_irradiance = 0;

    if (USE_CSGI && debug_shading_mode != SHADING_MODE_RTX_OFF) {
        if (USE_RTDGI) {
            gi_irradiance = rtdgi_tex[px].rgb;
        } else {
            // TODO: this could use bent normals to avoid leaks, or could be integrated into the SSAO loop,
            // Note: point-lookup doesn't leak, so multiple bounces should be fine
            float3 to_eye = get_eye_position() - pt_ws.xyz;
            float3 pseudo_bent_normal = normalize(normalize(to_eye) + gbuffer.normal);
            
            csgi_irradiance = lookup_csgi(
                pt_ws.xyz,
                gbuffer.normal,
                CsgiLookupParams::make_default()
                    //.with_sample_directional_radiance(gbuffer.normal)
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
        total_radiance += rtr_tex[px].xyz * brdf.energy_preservation.preintegrated_reflection;
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
        output = rtr_tex[px].xyz;
    }

    //const float3 bent_normal_dir = mul(frame_constants.view_constants.view_to_world, float4(ssgi.xyz, 0)).xyz;
    //output = pow((bent_normal_dir) * 0.5 + 0.5, 2);
    //output = bent_normal_dir * 0.5 + 0.5;
    //output = pow(gbuffer.normal.xyz * 0.5 + 0.5, 2);

    if (debug_shading_mode == SHADING_MODE_DIFFUSE_GI) {
        output = gi_irradiance;
    }

    //output = gbuffer.metalness;
    //output = gbuffer.roughness;
    //output = gbuffer.albedo;
    //output = ssgi.rgb;

    #if 0
        output = lookup_csgi(
            pt_ws.xyz,
            gbuffer.normal,
            CsgiLookupParams::make_default()
                .with_direct_light_only(true)
                //.with_sample_directional_radiance(gbuffer.normal)
                //.with_bent_normal(pseudo_bent_normal)
        );
    #endif

    output_tex[px] = float4(output, 1.0);
}
