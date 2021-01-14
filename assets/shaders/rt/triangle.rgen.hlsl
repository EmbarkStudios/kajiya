#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/rt.hlsl"

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
    const float3 ray_origin_ws = view_ray_context.ray_origin_ws();
    const float3 ray_dir_ws = view_ray_context.ray_dir_ws();

    RayDesc ray;
    ray.Origin = ray_origin_ws.xyz;
    ray.Direction = normalize(ray_dir_ws.xyz);
    ray.TMin = 0.0;
    ray.TMax = FLT_MAX;

    GbufferRayPayload payload = GbufferRayPayload::new_miss();

    TraceRay(acceleration_structure, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 0, 0, ray, payload);

    if (/*launchIndex.x < 1280 / 2 && */payload.is_hit()) {
        float3 hit_point = ray.Origin + ray.Direction * payload.t;

        RayDesc shadow_ray;
        shadow_ray.Origin = hit_point;
        shadow_ray.Direction = normalize(float3(1, 1, 1));
        shadow_ray.TMin = 1e-4;
        shadow_ray.TMax = 100000.0;

        ShadowRayPayload shadow_payload;
        shadow_payload.is_shadowed = true;
        TraceRay(
            acceleration_structure,
            RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            0xff, 0, 0, 0, shadow_ray, shadow_payload
        );

        float4 gbuffer = payload.gbuffer_packed;

        float3 albedo = unpack_color_888(asuint(gbuffer.x));
        float3 normal = unpack_normal_11_10_11(gbuffer.y);
        float roughness = sqrt(gbuffer.z);
        float metalness = gbuffer.w;

        float3 v = -normalize(ray_dir_ws.xyz);
        float3 l = normalize(float3(1, 1, 1));

        float3x3 shading_basis = build_orthonormal_basis(normal);

        SpecularBrdf specular_brdf;
        specular_brdf.roughness = roughness;
        specular_brdf.albedo = lerp(0.04, albedo, metalness);

        DiffuseBrdf diffuse_brdf;
        diffuse_brdf.albedo = max(0.0, 1.0 - metalness) * albedo;

        float3 wo = mul(v, shading_basis);
        float3 wi = mul(l, shading_basis);

        BrdfValue spec = specular_brdf.evaluate(wo, wi);
        BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

        float3 radiance = (spec.value() + spec.transmission_fraction * diff.value()) * max(0.0, wi.z);
        float3 ambient = ambient_light * albedo;

        float3 light_radiance = shadow_payload.is_shadowed ? 0.0 : 5.0;

        float4 res = float4(0.0.xxx, 1.0);
        res.xyz += radiance * light_radiance + ambient;
        res.xyz = neutral_tonemap(res.xyz);

        output_tex[launchIndex] = res;
    }
}
