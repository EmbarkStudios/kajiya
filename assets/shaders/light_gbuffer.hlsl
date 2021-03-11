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
#include "inc/atmosphere.hlsl"

#include "inc/hash.hlsl"
#include "inc/color.hlsl"

#define USE_SURFEL_GI 0
#define USE_SSGI 1
#define USE_CSGI 1
#define USE_RTR 1

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] Texture2D<float> sun_shadow_mask_tex;
[[vk::binding(3)]] Texture2D<float4> ssgi_tex;
[[vk::binding(4)]] Texture2D<float4> rtr_tex;
[[vk::binding(5)]] Texture2D<float4> base_light_tex;
[[vk::binding(6)]] RWTexture2D<float4> output_tex;
[[vk::binding(7)]] RWTexture2D<float4> debug_out_tex;
[[vk::binding(8)]] Texture3D<float4> csgi_cascade0_tex;
[[vk::binding(9)]] Texture3D<float4> csgi2_direct_tex;
[[vk::binding(10)]] Texture3D<float4> csgi2_indirect_tex;
[[vk::binding(11)]] cbuffer _ {
    float4 output_tex_size;
    float4 CSGI_SLICE_DIRS[16];
    float4 CSGI_SLICE_CENTERS[16];
};

#include "csgi/common.hlsl"
#include "csgi/lookup.hlsl"

#include "csgi2/common.hlsl"

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

    #if 0
        ruv.x *= output_tex_size.x / output_tex_size.y;
        output_tex[px] = bindless_textures[1].SampleLevel(sampler_lnc, uv, 0) * (all(uv == saturate(uv)) ? 1 : 0);
        return;
    #endif

    float4 gbuffer_packed = gbuffer_tex[px];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        float3 output = atmosphere_default(outgoing_ray.Direction, SUN_DIRECTION);
        output_tex[px] = float4(output, 1);
        debug_out_tex[px] = float4(output, 1);
        //output_tex[px] = float4(0.1.xxx, 1.0);
        return;
    }

    float z_over_w = depth_tex[px];
    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

    const float3 to_light_norm = SUN_DIRECTION;
    
    const float shadow_mask = sun_shadow_mask_tex[px];

    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();
    //gbuffer.roughness = 0.9;
    //gbuffer.metalness = 0;
    //gbuffer.albedo = 0.5;
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

    LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);
    const float3 brdf_value = brdf.evaluate_directional_light(wo, wi) * max(0.0, wi.z);
    const float3 light_radiance = shadow_mask * SUN_COLOR;
    float3 total_radiance = brdf_value * light_radiance;

    //const float3 radiance = (spec.value + spec.transmission_fraction * diff.value) * max(0.0, wi.z);
    //const float3 light_radiance = shadow_mask * SUN_COLOR;
    //total_radiance += radiance * light_radiance;

    //res.xyz += radiance * SUN_COLOR + ambient;
    //res.xyz += albedo;
    //res.xyz += metalness;
    //res.xyz = metalness;
    //res.xyz = 0.0001 / z_over_w;
    //res.xyz = frac(pt_ws.xyz * 10.0);
    //res.xyz = brdf_d * 0.1;

    //res.xyz = 1.0 - exp(-res.xyz);
    //total_radiance = preintegrated_specular_brdf_fg(specular_brdf.albedo, specular_brdf.roughness, wo.z);

    //uint pt_hash = hash3(asuint(int3(floor(pt_ws.xyz * 3.0))));
    //total_radiance += uint_id_to_color(pt_hash);

    float3 gi_irradiance = 0.0.xxx;

    #if USE_SURFEL_GI
        gi_irradiance = base_light_tex[px].xyz;
    #endif

    #if USE_CSGI
        gi_irradiance = lookup_csgi(pt_ws.xyz, gbuffer.normal, CsgiLookupParams::make_default());
    #endif

    #if USE_SSGI
        float4 ssgi = ssgi_tex[px];

        // HACK: need directionality in GI so that it can be properly masked.
        // If simply masking with the AO term, it tends to over-darken.
        // Reduce some of the occlusion, but for energy conservation, also reduce
        // the light added.
        const float4 biased_ssgi = lerp(ssgi, float4(0, 0, 0, 1), 0.3);

        total_radiance +=
            (gi_irradiance * biased_ssgi.a + biased_ssgi.rgb)
            * brdf.diffuse_brdf.albedo
            * brdf.energy_preservation.preintegrated_transmission_fraction
            ;
        // total_radiance = ssgi.a;
    #else
        total_radiance += gi_irradiance
            * brdf.diffuse_brdf.albedo
            * brdf.energy_preservation.preintegrated_transmission_fraction
            ;
    #endif

    //total_radiance = gbuffer.albedo * (ssgi.a + ssgi.rgb);

    #if USE_RTR
        total_radiance += rtr_tex[px].xyz * brdf.energy_preservation.preintegrated_reflection;
    #endif

    output_tex[px] = float4(total_radiance, 1.0);

    float3 debug_out = total_radiance;
        //base_light_tex[px].xyz * ssgi.a + ssgi.rgb,
        //base_light_tex[px].xyz,
        //ssgi.rgb,
        //lerp(ssgi.rgb, base_light_tex[px].xyz, ssgi.a) + ssgi.rgb,
    
    #if 0
        float4 pos_vs = mul(frame_constants.view_constants.world_to_view, pt_ws);
        const float view_dot = -normalize(pos_vs.xyz).z;

        float3 v_ws = normalize(mul(frame_constants.view_constants.view_to_world, float4(0, 0, -1, 0)).xyz);

        debug_out +=
            smoothstep(0.997, 1.0, view_dot) * gbuffer.albedo * max(0.0, dot(gbuffer.normal, -v_ws)) / M_PI;
    #endif

    //debug_out = rtr_tex[px].www * 0.2;
    //debug_out = rtr_tex[px].xyz;

    //debug_out = base_light_tex[px].xyz;
    
    //const float3 bent_normal_dir = mul(frame_constants.view_constants.view_to_world, float4(ssgi.xyz, 0)).xyz;
    //debug_out = pow((bent_normal_dir) * 0.5 + 0.5, 2);
    //debug_out = bent_normal_dir * 0.5 + 0.5;
    //debug_out = pow(gbuffer.normal.xyz * 0.5 + 0.5, 2);
    //debug_out = base_light_tex[px].xyz;

    //debug_out = gi_irradiance;
    //debug_out = gbuffer.albedo;

#if 1
    float3 total_gi = 0;
    float total_gi_wt = 0;

    //const uint gi_slice_idx = 3;
    for (uint gi_slice_idx = 0; gi_slice_idx < 6; ++gi_slice_idx) {
        const float3 volume_center = CSGI2_VOLUME_CENTER;
        const float normal_offset_scale = 1.0;
        const float3 vol_pos = (pt_ws.xyz - volume_center + gbuffer.normal * normal_offset_scale * CSGI2_VOXEL_SIZE);
        const int3 gi_vx = int3(vol_pos / CSGI2_VOXEL_SIZE + CSGI2_VOLUME_DIMS / 2);
        const float wt = saturate(0 + 1 * dot(CSGI2_SLICE_DIRS[gi_slice_idx], gbuffer.normal));
        total_gi += csgi2_indirect_tex[gi_vx + int3(CSGI2_VOLUME_DIMS * gi_slice_idx, 0, 0)].rgb * wt;
        total_gi_wt += wt;
    }
    debug_out = total_gi / total_gi_wt;
    //debug_out = frac(gi_vx / 64.0);
#endif

#if 0
    debug_out = bindless_textures[0][px / 16].rgb;
#endif

    debug_out_tex[px] = float4(debug_out, 1.0);
}
