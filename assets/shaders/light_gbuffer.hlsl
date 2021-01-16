#include "inc/frame_constants.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/brdf.hlsl"
#include "inc/uv.hlsl"
#include "inc/rt.hlsl"
#include "inc/tonemap.hlsl"

Texture2D<float4> gbuffer_tex;
Texture2D<float> depth_tex;
RWTexture2D<float4> output_tex;

#define PI 3.14159
#define TWO_PI 6.28318

static const float3 ambient_light = 0.1;
static const float3 SUN_COLOR = float3(1.0, 1.0, 1.0) * 30;

float g_smith_ggx_correlated(float ndotv, float ndotl, float ag) {
	float ag2 = ag * ag;

	float lambda_v = ndotl * sqrt((-ndotv * ag2 + ndotv) * ndotv + ag2);
	float lambda_l = ndotv * sqrt((-ndotl * ag2 + ndotl) * ndotl + ag2);

	return 2.0 * ndotl * ndotv / (lambda_v + lambda_l);
}

float d_ggx(float ndotm, float a) {
    float a2 = a * a;
	float denom_sqrt = ndotm * ndotm * (a2 - 1.0) + 1.0;
	return a2 / (PI * denom_sqrt * denom_sqrt);
}

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    float4 gbuffer_packed = gbuffer_tex[pix];
    if (all(gbuffer_packed == 0.0.xxxx)) {
        output_tex[pix] = float4(ambient_light, 1.0);
        return;
    }

    float4 output_tex_size = float4(1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0);
    float2 uv = get_uv(pix, output_tex_size);

    /*float z_over_w = depth_tex[pix];
    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(frame_constants.view_constants.view_to_world, mul(frame_constants.view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;*/

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
    const float3 to_light_norm = normalize(float3(1, 1, 1));
    
    // TODO
    const bool is_shadowed = false;

    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

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
    const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;

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
    total_radiance = neutral_tonemap(total_radiance);

    output_tex[pix] = float4(total_radiance, 1.0);
}
