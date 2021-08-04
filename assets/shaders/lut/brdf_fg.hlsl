#include "../inc/brdf.hlsl"
#include "../inc/quasi_random.hlsl"

[[vk::binding(0)]] RWTexture2D<float2> output_tex;

float2 integrate_brdf(float roughness, float ndotv) {
    float3 wo = float3(sqrt(1.0 - ndotv * ndotv), 0, ndotv);

    float a = 0;
    float b = 0;

    SpecularBrdf brdf_a;
    brdf_a.roughness = roughness;
    brdf_a.albedo = 1.0.xxx;

    SpecularBrdf brdf_b = brdf_a;
    brdf_b.albedo = 0.0;

    static const uint num_samples = 1024;
    for (uint i = 0; i < num_samples; ++i) {
        float2 urand = hammersley(i, num_samples);
        BrdfSample v_a = brdf_a.sample(wo, urand);

        if (v_a.is_valid()) {
            BrdfValue v_b = brdf_b.evaluate(wo, v_a.wi);

            a += (v_a.value_over_pdf.x - v_b.value_over_pdf.x);
            b += v_b.value_over_pdf.x;
        }
    }

    return float2(a, b) / num_samples;
}

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    float ndotv = (pix.x / (BRDF_FG_LUT_DIMS.x - 1.0)) * (1.0 - 1e-3) + 1e-3;
    float roughness = max(1e-5, pix.y / (BRDF_FG_LUT_DIMS.y - 1.0));

    output_tex[pix] = integrate_brdf(roughness, ndotv);
    //output_tex[pix] = float4(ndotv, roughness, 0, 1);
}
