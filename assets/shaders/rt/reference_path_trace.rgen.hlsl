#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/hash.hlsl"

static const uint MAX_PATH_LENGTH = 5;
static const float3 sun_color = float3(1.0, 1.0, 1.0) * 9;
static const bool FIREFLY_SUPPRESSION = true;

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;
[[vk::binding(0, 0)]] RWTexture2D<float4> output_tex;

float3 sample_environment_light(float3 dir) {
    float3 col = (dir.zyx * float3(1, 1, -1) * 0.5 + float3(0.6, 0.5, 0.5)) * 0.75;
    col = lerp(col, 1.3.xxx * calculate_luma(col), smoothstep(-0.2, 1.0, dir.y).xxx);
    return col;
}

// Approximate Gaussian remap
// https://www.shadertoy.com/view/MlVSzw
float inv_error_function(float x, float truncation) {
    static const float ALPHA = 0.14;
    static const float INV_ALPHA = 1.0 / ALPHA;
    static const float K = 2.0 / (M_PI * ALPHA);

	float y = log(max(truncation, 1.0 - x*x));
	float z = K + 0.5 * y;
	return sqrt(max(0.0, sqrt(z*z - y * INV_ALPHA) - z)) * sign(x);
}
float remap_unorm_to_gaussian(float x, float truncation) {
	return inv_error_function(x * 2.0 - 1.0, truncation);
}

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    uint seed = hash_combine2(hash_combine2(px.x, hash1(px.y)), frame_constants.frame_index);

    const float psf_scale = 0.85;
    float px_off0 = 0.5 + psf_scale * remap_unorm_to_gaussian(uint_to_u01_float(hash1_mut(seed)), 1e-5);
    float px_off1 = 0.5 + psf_scale * remap_unorm_to_gaussian(uint_to_u01_float(hash1_mut(seed)), 1e-5);

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

    [loop]
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
            specular_brdf.albedo = lerp(0.04, gbuffer.albedo, gbuffer.metalness);

            if (FIREFLY_SUPPRESSION) {
                specular_brdf.roughness = lerp(gbuffer.roughness, 1.0, 1.0-exp(-0.25 * path_length));
            } else {
                specular_brdf.roughness = gbuffer.roughness;
            }

            DiffuseBrdf diffuse_brdf;
            diffuse_brdf.albedo = max(0.0, 1.0 - gbuffer.metalness) * gbuffer.albedo;

            const BrdfValue spec = specular_brdf.evaluate(wo, wi);
            const BrdfValue diff = diffuse_brdf.evaluate(wo, wi);

            const float3 radiance = (spec.value() + spec.transmission_fraction * diff.value()) * max(0.0, wi.z);
            const float3 light_radiance = is_shadowed ? 0.0 : sun_color;

            //if (path_length > 0)
            {
                total_radiance += throughput * radiance * light_radiance;
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
            total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
            break;
        }
    }

    output_tex[px] += float4(total_radiance, 1.0);
}
