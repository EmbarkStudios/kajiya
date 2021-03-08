#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/gbuffer.hlsl"

static const float3 ambient_light = 0.1;

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] RWTexture2D<float4> output_tex;

[shader("raygeneration")]
void main()
{
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const float2 dims = DispatchRaysDimensions().xy;

    const float2 pixelCenter = launchIndex + 0.5;
    const float2 uv = pixelCenter / dims.xy;

    const ViewRayContext view_ray_context = ViewRayContext::from_uv(uv);
    const float3 ray_dir_ws = view_ray_context.ray_dir_ws();

    const RayDesc primary_ray = new_ray(
        view_ray_context.ray_origin_ws(), 
        normalize(ray_dir_ws.xyz),
        0.0,
        FLT_MAX
    );

    GbufferRayPayload payload = GbufferRayPayload::new_miss();
    TraceRay(acceleration_structure, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 0, 0, primary_ray, payload);

    if (/*launchIndex.x < 1280 / 2 && */payload.is_hit()) {
        const float3 hit_point = primary_ray.Origin + primary_ray.Direction * payload.t;
        const float3 to_light_norm = normalize(float3(1, 1, 1));
        
        const bool is_shadowed = rt_is_shadowed(
            acceleration_structure,
            new_ray(
                hit_point,
                normalize(float3(1, 1, 1)),
                1e-4,
                FLT_MAX
        ));

        const GbufferData gbuffer = payload.gbuffer_packed.unpack();

        const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
        const float3 wo = mul(-normalize(ray_dir_ws.xyz), shading_basis);
        const float3 wi = mul(to_light_norm, shading_basis);

        SpecularBrdf specular_brdf;
        specular_brdf.roughness = gbuffer.roughness;
        specular_brdf.albedo = lerp(0.04, gbuffer.albedo, gbuffer.metalness);

        DiffuseBrdf diffuse_brdf;
        diffuse_brdf.albedo = max(0.0, 1.0 - gbuffer.metalness) * gbuffer.albedo;

        const BrdfValue spec = specular_brdf.evaluate(wo, wi);
        const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

        const float3 radiance = (spec.value + spec.transmission_fraction * diff.value) * max(0.0, wi.z);
        const float3 ambient = ambient_light * gbuffer.albedo;

        const float3 light_radiance = is_shadowed ? 0.0 : 5.0;

        float4 res = float4(0.0.xxx, 1.0);
        res.xyz += radiance * light_radiance + ambient;
        res.xyz = neutral_tonemap(res.xyz);

        output_tex[launchIndex] = res;
    }
}
