#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/hash.hlsl"

static const uint MAX_PATH_LENGTH = 3;
static const float3 ambient_light = float3(0.02, 0.06, 0.1);

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] RWTexture2D<float4> output_tex;


[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    uint seed = hash_combine2(px.x, hash1(px.y));
    
    float px_off0 = uint_to_u01_float(hash1_mut(seed));
    float px_off1 = uint_to_u01_float(hash1_mut(seed));

    const float2 pixel_center = px + float2(px_off0, px_off1);
    const float2 uv = pixel_center / DispatchRaysDimensions().xy;

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

    float3 throughput = 1.0.xxx;
    float3 total_radiance = 0.0.xxx;

    for (uint path_length = 0; path_length < MAX_PATH_LENGTH; ++path_length) {
        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
        if (primary_hit.is_hit) {
            const float3 to_light_norm = normalize(float3(1, 1, 1));
            
            const bool is_shadowed = rt_is_shadowed(
                acceleration_structure,
                new_ray(
                    primary_hit.position,
                    normalize(float3(1, 1, 1)),
                    1e-4,
                    FLT_MAX
            ));

            const GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();

            const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
            const float3 wo = mul(-outgoing_ray.Direction, shading_basis);
            const float3 wi = mul(to_light_norm, shading_basis);

            SpecularBrdf specular_brdf;
            specular_brdf.roughness = gbuffer.roughness;
            specular_brdf.albedo = lerp(0.04, gbuffer.albedo, gbuffer.metalness);

            DiffuseBrdf diffuse_brdf;
            diffuse_brdf.albedo = max(0.0, 1.0 - gbuffer.metalness) * gbuffer.albedo;

            const BrdfValue spec = specular_brdf.evaluate(wo, wi);
            const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

            const float3 radiance = (spec.value() + spec.transmission_fraction * diff.value()) * max(0.0, wi.z);
            const float3 ambient = ambient_light * gbuffer.albedo;
            const float3 light_radiance = is_shadowed ? 0.0 : 5.0;

            //if (path_length > 0)
            {
                total_radiance += throughput * (radiance * light_radiance + ambient);
            }

            const float u0 = uint_to_u01_float(hash1_mut(seed));
            const float u1 = uint_to_u01_float(hash1_mut(seed));

            const float approx_fresnel = calculate_luma(eval_fresnel_schlick(specular_brdf.albedo, 1.0.xxx, wo.z));

            float spec_p = approx_fresnel;
            float diffuse_p = (1.0 - approx_fresnel) * calculate_luma(diffuse_brdf.albedo);
            const float layers_p_sum = diffuse_p + spec_p;
            
            diffuse_p /= layers_p_sum;
            spec_p /= layers_p_sum;

            BrdfSample brdf_sample;
            float lobe_pdf;

            const float lobe_xi = uint_to_u01_float(hash1_mut(seed));
            if (lobe_xi < spec_p) {
                brdf_sample = specular_brdf.sample(wo, float2(u0, u1));
                lobe_pdf = spec_p;
            } else {
                brdf_sample = diffuse_brdf.sample(wo, float2(u0, u1));
                lobe_pdf = diffuse_p;
            }

            if (lobe_pdf > 1e-9 && brdf_sample.is_valid()) {
                outgoing_ray.Origin = primary_hit.position;
                outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
                outgoing_ray.TMin = 1e-4;
                throughput *= brdf_sample.value_over_pdf / lobe_pdf;
            } else {
                break;
            }
        } else {
            total_radiance += throughput * ambient_light;
            break;
        }
    }

    output_tex[px] = float4(total_radiance, 1.0);
}
