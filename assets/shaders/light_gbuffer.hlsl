#include "inc/frame_constants.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/brdf.hlsl"
#include "inc/uv.hlsl"
#include "inc/rt.hlsl"
#include "inc/tonemap.hlsl"
#include "inc/bindless_textures.hlsl"
#include "inc/atmosphere.hlsl"

#include "inc/hash.hlsl"
#include "inc/color.hlsl"

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float> sun_shadow_mask_tex;
[[vk::binding(3)]] Texture2D<float4> ssgi_tex;
[[vk::binding(4)]] Texture2D<float4> base_light_tex;
[[vk::binding(5)]] RWTexture2D<float4> output_tex;
[[vk::binding(6)]] cbuffer _ {
    float4 output_tex_size;
};

SamplerState sampler_lnc;

static const float3 ambient_light = 0.0;
static const float3 SUN_DIRECTION = normalize(float3(1, 1.6, -0.2));
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

float3 preintegrated_specular_brdf_fg(float3 specular_albedo, float roughness, float ndotv) {
    float2 uv = float2(ndotv, roughness) * BRDF_FG_LUT_UV_SCALE + BRDF_FG_LUT_UV_BIAS;
    float2 fg = bindless_textures[0].SampleLevel(sampler_lnc, uv, 0).xy;
    #if 0
        return (specular_albedo * fg.x + fg.y);
    #else
        float3 single_scatter = specular_albedo * fg.x + fg.y;
        float energy_loss_per_bounce = 1.0 - (fg.x + fg.y);

        // Lost energy accounted for by an infinite geometric series:
        /*return single_scatter
            + energy_loss_per_bounce * single_scatter
            + energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * single_scatter
            + energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * single_scatter
            + energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * specular_albedo * energy_loss_per_bounce * single_scatter
            + ...
            ;*/

        // Closed-form solution:
        float3 bounce_radiance = energy_loss_per_bounce * specular_albedo;
        float3 albedo_inf_series = energy_loss_per_bounce * single_scatter / (1.0 - bounce_radiance);
        return single_scatter + albedo_inf_series;
    #endif
}

[numthreads(8, 8, 1)]
void main(in uint2 px : SV_DispatchThreadID) {
    float2 uv = get_uv(px, output_tex_size);

    #if 0
        ruv.x *= output_tex_size.x / output_tex_size.y;
        output_tex[px] = bindless_textures[1].SampleLevel(sampler_lnc, uv, 0) * (all(uv == saturate(uv)) ? 1 : 0);
        return;
    #endif

    float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        output_tex[px] = float4(neutral_tonemap(ambient_light), 1.0);
        return;
    }

    float z_over_w = depth_tex[px];
    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

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

    static const float3 throughput = 1.0.xxx;
    const float3 to_light_norm = SUN_DIRECTION;
    
    const float shadow_mask = sun_shadow_mask_tex[px];

    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
            //gbuffer.albedo = float3(1, 0.765557, 0.336057);
            //gbuffer.metalness = 1.0;
            //gbuffer.roughness = clamp((int(pt_ws.x * 0.2) % 5) / 5.0, 1e-4, 1.0);
    //gbuffer.roughness = 0.9;
    //gbuffer.metalness = 1;
    //gbuffer.albedo = 0.8;
    //gbuffer.metalness = 1;

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

    SpecularBrdf specular_brdf;
    specular_brdf.albedo = lerp(0.04, gbuffer.albedo, gbuffer.metalness);
    specular_brdf.roughness = gbuffer.roughness;

    DiffuseBrdf diffuse_brdf;
    diffuse_brdf.albedo = max(0.0, 1.0 - gbuffer.metalness) * gbuffer.albedo;

    const BrdfValue spec = specular_brdf.evaluate(wo, wi);
    const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

    const float3 radiance = (spec.value() + spec.transmission_fraction * diff.value()) * max(0.0, wi.z);
    const float3 light_radiance = shadow_mask * SUN_COLOR;

    float3 total_radiance = 0.0.xxx;
    total_radiance += throughput * radiance * light_radiance;
    total_radiance += ambient_light * gbuffer.albedo;

    //res.xyz += radiance * SUN_COLOR + ambient;
    //res.xyz += albedo;
    //res.xyz += metalness;
    //res.xyz = metalness;
    //res.xyz = 0.0001 / z_over_w;
    //res.xyz = frac(pt_ws.xyz * 10.0);
    //res.xyz = brdf_d * 0.1;

    //res.xyz = 1.0 - exp(-res.xyz);
    //total_radiance = preintegrated_specular_brdf_fg(specular_brdf.albedo, specular_brdf.roughness, wo.z) * ambient_light;

    //uint pt_hash = hash3(asuint(int3(floor(pt_ws.xyz * 3.0))));
    //total_radiance += uint_id_to_color(pt_hash);

    #if 1
        const float4 ssgi = ssgi_tex[px];
        total_radiance += (base_light_tex[px].xyz * ssgi.a + ssgi.rgb) * gbuffer.albedo;
        // total_radiance = ssgi.a;
    #else
        total_radiance += base_light_tex[px].xyz * gbuffer.albedo;
    #endif
    
    //total_radiance = neutral_tonemap(total_radiance);
    //total_radiance = gbuffer.metalness;

    output_tex[px] = float4(total_radiance, 1.0);
}
