#ifndef NOTORIOUS6_DISPLAY_TRANSFORM_HLSL
#define NOTORIOUS6_DISPLAY_TRANSFORM_HLSL

#include "ictcp.hlsl"
#include "luv.hlsl"
#include "oklab.hlsl"
#include "lab.hlsl"
#include "helmholtz_kohlrausch.hlsl"
#include "ycbcr.hlsl"
#include "ipt.hlsl"
#include "bezold_brucke.hlsl"

// The space to perform chroma attenuation in. More details in the `compress_stimulus` function.
// Oklab works well, but fails at pure blues.
// ICtCp seems to work pretty well all around.
#define PERCEPTUAL_SPACE_OKLAB 0
#define PERCEPTUAL_SPACE_ICTCP 1
#define PERCEPTUAL_SPACE_IPT 2
#define PERCEPTUAL_SPACE_NONE 3

// Brightness compression curves:
#define BRIGHTNESS_COMPRESSION_CURVE_REINHARD 0
#define BRIGHTNESS_COMPRESSION_CURVE_SIRAGUSANO_SMITH 1    // :P

// ----------------------------------------------------------------
// Configurable stuff:

#define BRIGHTNESS_COMPRESSION_CURVE BRIGHTNESS_COMPRESSION_CURVE_SIRAGUSANO_SMITH

// Choose the perceptual space for chroma attenuation.
#define PERCEPTUAL_SPACE PERCEPTUAL_SPACE_IPT

// Match target compressed brightness while attenuating chroma.
// Important in the low end, as well as at the high end of blue and red.
#define USE_BRIGHTNESS_LINEAR_CHROMA_ATTENUATION 1

// Controls for manual desaturation of lighter than "white" stimulus (greens, yellows);
// see comments in the code for more details.
#define CHROMA_ATTENUATION_START 0.0
#define CHROMA_ATTENUATION_EXPONENT_MIN 3.0
#define CHROMA_ATTENUATION_EXPONENT_MAX 4.0

// ----------------------------------------------------------------

#define USE_BEZOLD_BRUCKE_SHIFT 1
#define BEZOLD_BRUCKE_BRUTE_FORCE 0
#define BEZOLD_BRUCKE_SHIFT_RAMP 5
#define USE_LONG_TAILED_CHROMA_ATTENUATION 1
#define CHROMA_ATTENUATION_BIAS 1.03

// Based on the selection, define `linear_to_perceptual` and `perceptual_to_linear`
#if PERCEPTUAL_SPACE == PERCEPTUAL_SPACE_OKLAB
	#define linear_to_perceptual(col) sRGB_to_Oklab(col)
	#define perceptual_to_linear(col) Oklab_to_sRGB(col)
#elif PERCEPTUAL_SPACE == PERCEPTUAL_SPACE_ICTCP
	#define linear_to_perceptual(col) BT709_to_ICtCp(col)
	#define perceptual_to_linear(col) ICtCp_to_BT709(col)
#elif PERCEPTUAL_SPACE == PERCEPTUAL_SPACE_IPT
	#define linear_to_perceptual(col) XYZ_to_IPT(sRGB_to_XYZ(col))
	#define perceptual_to_linear(col) XYZ_to_sRGB(IPT_to_XYZ(col))
#elif PERCEPTUAL_SPACE == PERCEPTUAL_SPACE_NONE
	#define linear_to_perceptual(col) (col)
	#define perceptual_to_linear(col) (col)
#endif

// Map brightness through a curve yielding values in 0..1, working with linear stimulus values.
float compress_luminance(float v) {
	#if BRIGHTNESS_COMPRESSION_CURVE == BRIGHTNESS_COMPRESSION_CURVE_REINHARD
		// Reinhard
        const float k = 1.0;
		return pow(pow(v, k) / (pow(v, k) + 1.0), 1.0 / k);
	#elif BRIGHTNESS_COMPRESSION_CURVE == BRIGHTNESS_COMPRESSION_CURVE_SIRAGUSANO_SMITH
		// From Jed Smith: https://github.com/jedypod/open-display-transform/wiki/tech_tonescale,
        // based on stuff from Daniele Siragusano: https://community.acescentral.com/t/output-transform-tone-scale/3498/14
        // Reinhard with flare compensation.
        const float sx = 1.0;
        const float p = 1.2;
        const float sy = 1.0205;
		return saturate(sy * pow(v / (v + sx), p));
    #endif
}

float3 display_transform_sRGB(float3 input_stimulus) {
    if (USE_BEZOLD_BRUCKE_SHIFT) {
        const float t = sRGB_to_luminance(input_stimulus) / BEZOLD_BRUCKE_SHIFT_RAMP;
        const float shift_amount = t / (t + 1.0);

        #if BEZOLD_BRUCKE_BRUTE_FORCE
            float3 stimulus = XYZ_to_sRGB(bezold_brucke_shift_XYZ_brute_force(sRGB_to_XYZ(input_stimulus), shift_amount));
        #else
            float3 stimulus = XYZ_to_sRGB(bezold_brucke_shift_XYZ_with_lut(sRGB_to_XYZ(input_stimulus), shift_amount));
        #endif
        
        input_stimulus = stimulus;
    }

    const HelmholtzKohlrauschEffect hk = hk_from_sRGB(input_stimulus);

    // Find the shader_input luminance adjusted by the Helmholtz-Kohlrausch effect.
    const float input_equiv_lum = srgb_to_equivalent_luminance(hk, input_stimulus);

    // The highest displayable intensity stimulus with the same chromaticity as the shader_input,
    // and its associated equivalent luminance.
    const float3 max_intensity_rgb = input_stimulus / max(input_stimulus.r, max(input_stimulus.g, input_stimulus.b)).xxx;
    float max_intensity_equiv_lum = srgb_to_equivalent_luminance(hk, max_intensity_rgb);
    //return max_intensity_equiv_lum.xxx - 1.0;
    //return saturate(max_intensity_rgb);

    const float max_output_scale = 1.0;

    // Compress the brightness. We will then adjust the chromatic shader_input stimulus to match this.
    // Note that this is not the non-linear "L*", but a 0..`max_output_scale` value as a multilpier
    // over the maximum achromatic luminance.
    const float compressed_achromatic_luminance = compress_luminance(input_equiv_lum / max_output_scale) * max_output_scale;
    //const float compressed_achromatic_luminance = smoothstep(0.1, 0.9, shader_input.uv.x);

    // Scale the chromatic stimulus so that its luminance matches `compressed_achromatic_luminance`.
    // TODO: Overly simplistic, and does not accurately map the brightness.
    //
    // This will create (mostly) matching brightness, but potentially out of gamut components.
    float3 compressed_rgb = (max_intensity_rgb / max_intensity_equiv_lum) * compressed_achromatic_luminance;

    // The achromatic stimulus we'll interpolate towards to fix out-of-gamut stimulus.
    const float clamped_compressed_achromatic_luminance = min(1.0, compressed_achromatic_luminance);

    // We now want to map the out-of-gamut stimulus back to what our device can display.
    // Since both the `compressed_rgb` and `clamped_compressed_achromatic_luminance` are of the same-ish
    // brightness, and `clamped_compressed_achromatic_luminance.xxx` is guaranteed to be inside the gamut,
    // we can trace a path from `compressed_rgb` towards `clamped_compressed_achromatic_luminance.xxx`,
    // and stop once we have intersected the target gamut.

    // This has the effect of removing chromatic content from the compressed stimulus,
    // and replacing that with achromatic content. If we do that naively, we run into
    // a perceptual hue shift due to the Abney effect.
    //
    // To counter, we first transform both vertices of the path we want to trace
    // into a perceptual space which preserves sensation of hue, then we trace
    // a straight line _inside that space_ until we intersect the gamut.

	const float3 perceptual = linear_to_perceptual(compressed_rgb);
	const float3 perceptual_white = linear_to_perceptual(clamped_compressed_achromatic_luminance.xxx);

    // Values lighter than "white" are already within the gamut, so our brightness compression is "done".
    // Perceptually they look wrong though, as they don't follow the desaturation that other stimulus does.
    // We fix that manually here by biasing the interpolation towards "white" at the end of the brightness range.
    // This "fixes" the yellows and greens.
    
    // We'll make the transition towards white smoother in areas of high chromatic strength.
    //float chroma_strength = length(sRGB_to_YCbCr(max_intensity_rgb).yz);
    float chroma_strength = LAB_to_Lch(XYZ_to_LAB(sRGB_to_XYZ(max_intensity_rgb))).y / 100.0 * 0.4;
    //float chroma_strength = 1;

    const float chroma_attenuation_start = CHROMA_ATTENUATION_START;
    const float chroma_attenuation_exponent = lerp(CHROMA_ATTENUATION_EXPONENT_MAX, CHROMA_ATTENUATION_EXPONENT_MIN, chroma_strength);
    const float chroma_attenuation_t = saturate(
        (compressed_achromatic_luminance - min(1, max_intensity_equiv_lum) * chroma_attenuation_start)
        / ((CHROMA_ATTENUATION_BIAS * max_output_scale - min(1, max_intensity_equiv_lum) * chroma_attenuation_start))
    );

#if USE_LONG_TAILED_CHROMA_ATTENUATION
    float chroma_attenuation = asin(pow(chroma_attenuation_t, 3.0)) / M_PI * 2;
    
    // Window this with a soft falloff
    {
        const float compressed_achromatic_luminance2 = compress_luminance(0.125 * input_equiv_lum / max_output_scale) * max_output_scale;
        const float chroma_attenuation_t2 = saturate(
            (compressed_achromatic_luminance2 - min(1, max_intensity_equiv_lum) * 0.5)
            / ((max_output_scale - min(1, max_intensity_equiv_lum) * 0.5))
        );

        chroma_attenuation = lerp(chroma_attenuation, 1.0,
            1.0 - saturate(1.0 - pow(chroma_attenuation_t2, 4))
        );
    }
#else
    const float chroma_attenuation = pow(chroma_attenuation_t, chroma_attenuation_exponent);
#endif

    {
		const float3 perceptual_mid = lerp(perceptual, perceptual_white, chroma_attenuation);
		compressed_rgb = perceptual_to_linear(perceptual_mid);

        const HelmholtzKohlrauschEffect hk = hk_from_sRGB(compressed_rgb);

        #if USE_BRIGHTNESS_LINEAR_CHROMA_ATTENUATION
            for (int i = 0; i < 2; ++i) {
                const float current_brightness = srgb_to_equivalent_luminance(hk, compressed_rgb);
                compressed_rgb *= compressed_achromatic_luminance / max(1e-10, current_brightness);
            }
        #endif
    }

    // At this stage we still have out of gamut colors.
    // This takes a silly twist now. So far we've been careful to preserve hue...
    // Now we're going to let the channels clip, but apply a per-channel roll-off.
    // This sacrificies hue accuracy and brightness to retain saturation.

    if (true) {
        compressed_rgb = max(compressed_rgb, 0.0.xxx);

        const float p = 12.0;
        compressed_rgb = compressed_rgb * pow(pow(compressed_rgb, p.xxx) + 1.0, -1.0 / p.xxx);

        const float max_comp = max(compressed_rgb.r, max(compressed_rgb.g, compressed_rgb.b));
        const float max_comp_dist = max(max_comp - compressed_rgb.r, max(max_comp - compressed_rgb.g, max_comp - compressed_rgb.b));

        // Rescale so we can reach 100% white. Avoid rescaling very highly saturated colors,
        // as that would reintroduce discontinuities.
        compressed_rgb /= pow(lerp(0.5, 1.0, max_comp_dist), 1.0 / p);
    }

    //return hk_equivalent_luminance(compressed_rgb).xxx;
    //return compressed_achromatic_luminance.xxx;

    return compressed_rgb;
}

#endif  // NOTORIOUS6_DISPLAY_TRANSFORM_HLSL
