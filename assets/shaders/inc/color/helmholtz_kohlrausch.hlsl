#ifndef NOTORIOUS6_HELMHOLTZ_KOHLRAUSCH_HLSL
#define NOTORIOUS6_HELMHOLTZ_KOHLRAUSCH_HLSL

#include "math.hlsl"
#include "srgb.hlsl"

// Helmholtz-Kohlrausch adjustment methods
#define HK_ADJUSTMENT_METHOD_NONE 0
#define HK_ADJUSTMENT_METHOD_NAYATANI 1
#define HK_ADJUSTMENT_METHOD_CUSTOM_G0 2

// Adapting luminance (L_a) used for the H-K adjustment. 20 cd/m2 was used in Sanders and Wyszecki (1964)
#define HK_ADAPTING_LUMINANCE 20

// Choose the method for performing the H-K adjustment
#ifndef HK_ADJUSTMENT_METHOD
#define HK_ADJUSTMENT_METHOD HK_ADJUSTMENT_METHOD_CUSTOM_G0
#endif


// Helmholtz-Kohlrausch
// From https://github.com/ilia3101/HKE
// Based on Nayatani, Y. (1997). Simple estimation methods for the Helmholtz-Kohlrausch effect

// `uv`: CIE LUV u' and v'
// `adapt_lum`: adating luminance (L_a)
float hk_lightness_adjustment_multiplier_nayatani(float2 uv, float adapt_lum) {
    const float2 d65_uv = CIE_xyY_xy_to_LUV_uv(white_D65_xy);
    const float u_white = d65_uv[0];
    const float v_white = d65_uv[1];

    uv -= float2(u_white, v_white);

    float theta = atan2(uv[1], uv[0]);

    float q =
        - 0.01585
        - 0.03016 * cos(theta) - 0.04556 * cos(2 * theta)
        - 0.02667 * cos(3 * theta) - 0.00295 * cos(4 * theta)
        + 0.14592 * sin(theta) + 0.05084 * sin(2 * theta)
        - 0.01900 * sin(3 * theta) - 0.00764 * sin(4 * theta);

    float kbr = 0.2717 * (6.469 + 6.362 * pow(adapt_lum, 0.4495)) / (6.469 + pow(adapt_lum, 0.4495));
    float suv = 13.0 * length(uv);

    return 1.0 + (-0.1340 * q + 0.0872 * kbr) * suv;
}

// Heavily modified from Nayatani
// `uv`: CIE LUV u' and v'
float XYZ_to_hk_luminance_multiplier_custom_g0(float3 XYZ) {
    float2 uv = CIE_XYZ_to_LUV_uv(XYZ);
    const float2 d65_uv = CIE_xyY_xy_to_LUV_uv(white_D65_xy);
    const float u_white = d65_uv[0];
    const float v_white = d65_uv[1];
    uv -= float2(u_white, v_white);

    const float theta = atan2(uv[1], uv[0]);

    // ----
    // Custom q function eyeballed to achieve the greyness boundary condition in sRGB color sweeps.
    const uint SAMPLE_COUNT = 16;
    static const float samples[] = {
        -0.006,    // greenish cyan
        -0.021,    // greenish cyan
        -0.033,    // cyan
        -0.009,    // cyan
        0.14,    // blue
        0.114,   // purplish-blue
        0.111,   // magenta
        0.1005,   // magenta
        0.069,   // purplish-red
        0.0135,   // red
        -0.045,    // orange
        -0.075,    // reddish-yellow
        -0.075,    // yellow
        -0.03,   // yellowish-green
        0.006,   // green
        0.006,  // green
    };

    const float t = (theta / M_PI) * 0.5 + 0.5;
    const uint i0 = uint(floor(t * SAMPLE_COUNT)) % SAMPLE_COUNT;
    const uint i1 = (i0 + 1) % SAMPLE_COUNT;
    const float q0 = samples[(i0 + SAMPLE_COUNT - 1) % SAMPLE_COUNT];
    const float q1 = samples[i0];
    const float q2 = samples[i1];
    const float q3 = samples[(i1 + 1) % SAMPLE_COUNT];
    const float interp = (t - float(i0) / SAMPLE_COUNT) * SAMPLE_COUNT;
    const float q = catmull_rom(interp, q0, q1, q2, q3);
    // ----

    const float adapt_lum = 20.0;
    const float kbr = 0.2717 * (6.469 + 6.362 * pow(adapt_lum, 0.4495)) / (6.469 + pow(adapt_lum, 0.4495));
    const float suv = 13.0 * length(uv);

    // Nayatani scales _lightness_ which is approximately a cubic root of luminance
    // To scale luminance, we need the third power of that multiplier.
    const float mult_cbrt = 1.0 + (q + 0.0872 * kbr) * suv;
    return mult_cbrt * mult_cbrt * mult_cbrt;
}

struct HelmholtzKohlrauschEffect {
#if HK_ADJUSTMENT_METHOD == HK_ADJUSTMENT_METHOD_CUSTOM_G0
    float mult;
#else
    float omg_glsl_y_u_no_empty_struct;
#endif
};

HelmholtzKohlrauschEffect hk_from_sRGB(float3 stimulus) {
    HelmholtzKohlrauschEffect res;
    #if HK_ADJUSTMENT_METHOD == HK_ADJUSTMENT_METHOD_CUSTOM_G0
        res.mult = XYZ_to_hk_luminance_multiplier_custom_g0(sRGB_to_XYZ(stimulus));
    #endif
    return res;
}

float srgb_to_equivalent_luminance(HelmholtzKohlrauschEffect hk, float3 stimulus) {
    #if HK_ADJUSTMENT_METHOD == HK_ADJUSTMENT_METHOD_NAYATANI
        const float luminance = sRGB_to_luminance(stimulus);
        const float2 uv = CIE_XYZ_to_LUV_uv(sRGB_to_XYZ(stimulus));
        const float luv_brightness = luminance_to_LUV_L(luminance);
        const float mult = hk_lightness_adjustment_multiplier_nayatani(uv, HK_ADAPTING_LUMINANCE);
        return LUV_L_to_luminance(luv_brightness * mult);
    #elif HK_ADJUSTMENT_METHOD == HK_ADJUSTMENT_METHOD_CUSTOM_G0
        return hk.mult * sRGB_to_XYZ(stimulus).y;
    #elif HK_ADJUSTMENT_METHOD == HK_ADJUSTMENT_METHOD_NONE
        return sRGB_to_luminance(stimulus);
    #endif
}

#endif  // NOTORIOUS6_HELMHOLTZ_KOHLRAUSCH_HLSL
