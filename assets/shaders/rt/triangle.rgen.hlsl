#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/brdf.hlsl"

static const float3 ambient_light = 0.1;

float g_smith_ggx_correlated(float ndotv, float ndotl, float ag) {
	float ag2 = ag * ag;

	float lambda_v = ndotl * sqrt((-ndotv * ag2 + ndotv) * ndotv + ag2);
	float lambda_l = ndotv * sqrt((-ndotl * ag2 + ndotl) * ndotl + ag2);

	return 2.0 * ndotl * ndotv / (lambda_v + lambda_l);
}

float d_ggx(float ndotm, float a) {
    float a2 = a * a;
	float denom_sqrt = ndotm * ndotm * (a2 - 1.0) + 1.0;
	return a2 / (M_PI * denom_sqrt * denom_sqrt);
}

struct Payload {
    float4 gbuffer_packed;
    float t;
};

struct ShadowPayload {
    bool is_shadowed;
};


struct Attribute {
    float2 bary;
};

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] RWTexture2D<float4> output_tex;

[shader("raygeneration")]
void main()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = DispatchRaysDimensions().xy;

    float2 pixelCenter = launchIndex + 0.5;
    float2 uv = pixelCenter / dims.xy;

    ViewConstants view_constants = frame_constants.view_constants;
    float4 ray_dir_cs = float4(uv_to_cs(uv), 0.0, 1.0);
    float4 ray_dir_vs = mul(view_constants.sample_to_view, ray_dir_cs);
    float4 ray_dir_ws = mul(view_constants.view_to_world, ray_dir_vs);

    float4 ray_origin_cs = float4(uv_to_cs(uv), 1.0, 1.0);
    float4 ray_origin_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, ray_origin_cs));
    ray_origin_ws /= ray_origin_ws.w;

    /*float2 d = uv * 2.0 - 1.0;
    float aspectRatio = float(dims.x) / float(dims.y);*/

    RayDesc ray;
    ray.Origin = ray_origin_ws.xyz;
    ray.Direction = normalize(ray_dir_ws.xyz);
    ray.TMin = 0.001;
    ray.TMax = 100000.0;

    Payload payload;
    payload.gbuffer_packed = 0;
    payload.t = 0;

    TraceRay(acceleration_structure, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 0, 0, ray, payload);

    if (/*launchIndex.x < 1280 / 2 && */payload.t > 0.0) {
        float3 hit_point = ray.Origin + ray.Direction * payload.t;

        RayDesc shadow_ray;
        shadow_ray.Origin = hit_point;
        shadow_ray.Direction = normalize(float3(1, 1, 1));
        shadow_ray.TMin = 1e-4;
        shadow_ray.TMax = 100000.0;

        ShadowPayload shadow_payload;
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
