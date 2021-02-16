#include "../inc/uv.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/tonemap.hlsl"
#include "../inc/gbuffer.hlsl"
#include "../inc/brdf.hlsl"
#include "../inc/brdf_lut.hlsl"
#include "../inc/layered_brdf.hlsl"
#include "../inc/hash.hlsl"
#include "../inc/rt.hlsl"
#include "../inc/sun.hlsl"

#include "../inc/atmosphere.hlsl"
static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

#define USE_TEMPORAL_JITTER 1
#define USE_HEAVY_BIAS 0

// Strongly reduces roughness of secondary hits
#define USE_AGGRESSIVE_ROUGHNESS_BIAS 0

// BRDF bias
#define SAMPLING_BIAS 0.0


#if USE_HEAVY_BIAS
    #undef USE_AGGRESSIVE_ROUGHNESS_BIAS
    #define USE_AGGRESSIVE_ROUGHNESS_BIAS 1

    #undef SAMPLING_BIAS
    #define SAMPLING_BIAS 0.1
#endif

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

[[vk::binding(0)]] Texture2D<float4> gbuffer_tex;
[[vk::binding(1)]] Texture2D<float> depth_tex;
[[vk::binding(2)]] StructuredBuffer<uint> ranking_tile_buf;
[[vk::binding(3)]] StructuredBuffer<uint> scambling_tile_buf;
[[vk::binding(4)]] StructuredBuffer<uint> sobol_buf;
[[vk::binding(5)]] RWTexture2D<float4> out0_tex;
[[vk::binding(6)]] RWTexture2D<float4> out1_tex;
[[vk::binding(7)]] cbuffer _ {
    float4 gbuffer_tex_size;
};

float blue_noise_sampler(int pixel_i, int pixel_j, int sampleIndex, int sampleDimension)
{
	// wrap arguments
	pixel_i = pixel_i & 127;
	pixel_j = pixel_j & 127;
	sampleIndex = sampleIndex & 255;
	sampleDimension = sampleDimension & 255;

	// xor index based on optimized ranking
	// jb: 1spp blue noise has all 0 in ranking_tile_buf so we can skip the load
	int rankedSampleIndex = sampleIndex ^ ranking_tile_buf[sampleDimension + (pixel_i + pixel_j*128)*8];

	// fetch value in sequence
	int value = sobol_buf[sampleDimension + rankedSampleIndex*256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ scambling_tile_buf[(sampleDimension%8) + (pixel_i + pixel_j*128)*8];

	// convert to float and return
	float v = (0.5f+value)/256.0f;
	return v;
}

static const float SKY_DIST = 1e5;

[shader("raygeneration")]
void main() {
    const uint2 px = DispatchRaysIndex().xy;
    const uint2 hi_px = px * 2;
    float depth = depth_tex[hi_px];

    if (0.0 == depth) {
        out0_tex[px] = float4(0.0.xxx, -SKY_DIST);
        return;
    }

    const float2 uv = get_uv(hi_px, gbuffer_tex_size);
    const ViewRayContext view_ray_context = ViewRayContext::from_uv_and_depth(uv, depth);

    float4 gbuffer_packed = gbuffer_tex[hi_px];
    GbufferData gbuffer = GbufferDataPacked::from_uint4(asuint(gbuffer_packed)).unpack();

    const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
    const float3 ray_dir_ws = normalize(view_ray_context.ray_dir_ws_h.xyz);
    float3 wo = mul(-ray_dir_ws, shading_basis);

    const float3 ray_hit_ws = view_ray_context.ray_hit_ws();
    const float3 ray_hit_vs = view_ray_context.ray_hit_vs();
    const float3 refl_ray_origin = ray_hit_ws - ray_dir_ws * (length(ray_hit_vs) + length(ray_hit_ws)) * 1e-4;

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

#if USE_AGGRESSIVE_ROUGHNESS_BIAS
    const float roughness_bias = lerp(gbuffer.roughness, 1.0, 0.333);
#else
    const float roughness_bias = 0.5 * gbuffer.roughness;
#endif

    const uint seed = USE_TEMPORAL_JITTER ? frame_constants.frame_index : 0;
    uint rng = hash_combine2(hash_combine2(px.x, hash1(px.y)), seed);

#if 1
    const uint noise_offset = frame_constants.frame_index * (USE_TEMPORAL_JITTER ? 1 : 0);

    float2 urand = float2(
        blue_noise_sampler(px.x, px.y, noise_offset, 0),
        blue_noise_sampler(px.x, px.y, noise_offset, 1)
    );
#elif 0
    // 256x256 blue noise

    const uint noise_offset = frame_constants.frame_index * (USE_TEMPORAL_JITTER ? 1 : 0);

    float2 urand = bindless_textures[1][
        (px + int2(noise_offset * 59, noise_offset * 37)) & 255
    ].xy * 255.0 / 256.0 + 0.5 / 256.0;
#else
    float2 urand = float2(
        uint_to_u01_float(hash1_mut(rng)),
        uint_to_u01_float(hash1_mut(rng))
    );
#endif

    const float sampling_bias = SAMPLING_BIAS;
    urand.x = lerp(urand.x, 0.0, sampling_bias);

    BrdfSample brdf_sample = specular_brdf.sample(wo, urand);
    
#if USE_TEMPORAL_JITTER
    [loop]
    for (uint retry_count = 0; !brdf_sample.is_valid() && retry_count < 4; ++retry_count) {
        urand = float2(
            uint_to_u01_float(hash1_mut(rng)),
            uint_to_u01_float(hash1_mut(rng))
        );
        urand.x = lerp(urand.x, 0.0, sampling_bias);

        brdf_sample = specular_brdf.sample(wo, urand);
    }
#endif

    if (brdf_sample.is_valid()) {
        RayDesc outgoing_ray;
        outgoing_ray.Direction = mul(shading_basis, brdf_sample.wi);
        outgoing_ray.Origin = refl_ray_origin;
        outgoing_ray.TMin = 0;
        outgoing_ray.TMax = SKY_DIST;

        out1_tex[px] = float4(outgoing_ray.Direction, clamp(brdf_sample.pdf, 1e-5, 1e5));

        const GbufferPathVertex primary_hit = rt_trace_gbuffer(acceleration_structure, outgoing_ray);
        if (primary_hit.is_hit) {
            float3 total_radiance = 0.0.xxx;
            {
                const float3 to_light_norm = SUN_DIRECTION;
                const bool is_shadowed =
                    rt_is_shadowed(
                        acceleration_structure,
                        new_ray(
                            primary_hit.position,
                            to_light_norm,
                            1e-4,
                            SKY_DIST
                    ));

                GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
                gbuffer.roughness = lerp(gbuffer.roughness, 1.0, roughness_bias);
                const float3x3 shading_basis = build_orthonormal_basis(gbuffer.normal);
                const float3 wi = mul(to_light_norm, shading_basis);
                float3 wo = mul(-outgoing_ray.Direction, shading_basis);

                LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z);

                const float3 brdf_value = brdf.evaluate(wo, wi);
                const float3 light_radiance = is_shadowed ? 0.0 : SUN_COLOR;
                total_radiance += brdf_value * light_radiance;
            }

            out0_tex[px] = float4(total_radiance, primary_hit.ray_t);
        } else {
            out0_tex[px] = float4(0.0.xxx, SKY_DIST);
            //out0_tex[px] = float4(0.5.xxx, SKY_DIST);
            //out0_tex[px] = float4(atmosphere_default(outgoing_ray.Direction, SUN_DIRECTION), SKY_DIST);
        }
    } else {
        out0_tex[px] = float4(0.0.xxx, -SKY_DIST);
    }
}
