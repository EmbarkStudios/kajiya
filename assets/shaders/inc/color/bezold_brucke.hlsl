#include "xyz.hlsl"
#include "standard_observer.hlsl"

// Assigns uniform angles to hue in CIE 1931, however that doesn't mean much.
#define BB_LUT_LUT_MAPPING_ANGULAR 0

// Probably the best bang/buck.
#define BB_LUT_LUT_MAPPING_QUAD 1

// Not really an improvement.
#define BB_LUT_LUT_MAPPING_ROTATED_QUAD 2
#define BB_LUT_LUT_MAPPING_ROTATED_QUAD_ANGLE 0.8595

// Select the encoding method
#define BB_LUT_LUT_MAPPING BB_LUT_LUT_MAPPING_QUAD


float bb_xy_white_offset_to_lut_coord(float2 offset) {
    #if BB_LUT_LUT_MAPPING == BB_LUT_LUT_MAPPING_ANGULAR
        return frac((atan2(offset.y, offset.x) / M_PI) * 0.5);
    #elif BB_LUT_LUT_MAPPING == BB_LUT_LUT_MAPPING_QUAD
        offset /= max(abs(offset.x), abs(offset.y));
        float sgn = select((offset.x + offset.y) > 0.0, 1.0, -1.0);
        // NOTE: needs a `frac` if the sampler's U wrap mode is not REPEAT.
        return sgn * (0.125 * (offset.x - offset.y) + 0.25);
    #elif BB_LUT_LUT_MAPPING == BB_LUT_LUT_MAPPING_ROTATED_QUAD
        const float angle = BB_LUT_LUT_MAPPING_ROTATED_QUAD_ANGLE;
        offset = mul(float2x2(cos(angle), sin(angle), -sin(angle), cos(angle)), offset);
        offset /= max(abs(offset.x), abs(offset.y));
        float sgn = select((offset.x + offset.y) > 0.0, 1.0, -1.0);
        // NOTE: needs a `frac` if the sampler's U wrap mode is not REPEAT.
        return sgn * (0.125 * (offset.x - offset.y) + 0.25);
    #endif
}

float2 bb_lut_coord_to_xy_white_offset(float coord) {
    #if BB_LUT_LUT_MAPPING == BB_LUT_LUT_MAPPING_ANGULAR
        const float theta = coord * M_PI * 2.0;
        return float2(cos(theta), sin(theta));
    #elif BB_LUT_LUT_MAPPING == BB_LUT_LUT_MAPPING_QUAD
        float side = select(coord < 0.5, 1.0, -1.0);
        float t = frac(coord * 2);
        return side * normalize(
            lerp(float2(-1, 1), float2(1, -1), t)
            + lerp(float2(0, 0), float2(1, 1), 1 - abs(t - 0.5) * 2)
        );
    #elif BB_LUT_LUT_MAPPING == BB_LUT_LUT_MAPPING_ROTATED_QUAD
        float side = select(coord < 0.5, 1.0, -1.0);
        float t = frac(coord * 2);
        float2 offset = side * normalize(lerp(float2(-1, 1), float2(1, -1), t) + lerp(float2(0, 0), float2(1, 1), 1 - abs(t - 0.5) * 2));
        const float angle = BB_LUT_LUT_MAPPING_ROTATED_QUAD_ANGLE;
        return mul(offset, float2x2(cos(angle), sin(angle), -sin(angle), cos(angle)));
    #endif
}

float XYZ_to_BB_shift_nm(float3 XYZ) {
    const float2 white = white_D65_xy;

    const float2 xy = CIE_XYZ_to_xyY(XYZ).xy;
    float2 white_offset = xy - white_D65_xy;
    float theta = atan2(white_offset.y, white_offset.x);

    // Piece-wise linear match to Pridmore's plot for 10:100 cd/m^2
    const uint SAMPLE_COUNT = 26;
    static const float2 samples[] = {
        float2(0.0, 0),
        float2(0.084, -5.0),
        float2(0.152, -5.0),
        float2(0.2055, -4.0),
        float2(0.25, 0.0),
        float2(0.265, 2.3),
        float2(0.291, 5),
        float2(0.31, 6),
        float2(0.3285, 6.5),
        float2(0.356, 5.4),
        float2(0.395, 4.4),
        float2(0.4445, 3.93),
        float2(0.551, -4.9),
        float2(0.585, -6),
        float2(0.6065, -6),
        float2(0.6133, -3),
        float2(0.621, 1.42),
        float2(0.6245, 1.9),
        float2(0.633, 2.55),
        // non-spectral gap
        float2(0.92495, 2.55),
        float2(0.92525, 3.35),
        float2(0.9267, 4.8),
        float2(0.93, 6.15),
        float2(0.934, 7),
        float2(0.942, 5.95),
        float2(0.956, 4.0),
    };

    const float t = frac((-theta / M_PI) * 0.5 + 0.61);

    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        float2 p0 = samples[i];
        float2 p1 = samples[(i + 1) % SAMPLE_COUNT];
        float interp = (t - p0.x) / frac(p1.x - p0.x + 1);
        if (t >= p0.x && interp <= 1.0) {
            return lerp(p0.y, p1.y, interp);
        }
    }

    return 0.0;
}

// Apply Bezold–Brucke shift to XYZ stimulus. Loosely based on
// "Pridmore, R. W. (1999). Bezold–Brucke hue-shift as functions of luminance level,
// luminance ratio, interstimulus interval and adapting white for aperture and object colors.
// Vision Research, 39(23), 3873–3891. doi:10.1016/s0042-6989(99)00085-1"
float3 bezold_brucke_shift_XYZ_brute_force(float3 XYZ, float amount) {
    const float3 xyY = CIE_XYZ_to_xyY(XYZ);
    const float white_offset_magnitude = length(xyY.xy - white_D65_xy);
    const float bb_shift = XYZ_to_BB_shift_nm(XYZ);

    const float dominant_wavelength = CIE_xy_to_dominant_wavelength(xyY.xy);
    if (dominant_wavelength == -1) {
        // Non-spectral stimulus.
        // We could calculate the shift for the two corner vertices of the gamut,
        // then interpolate the shift between them, however the wavelengths
        // get so compressed in the xyY space near limits of vision, that
        // the shift is effectively nullified.
        return XYZ;
    }

    float3 shifted_xyY = wavelength_to_xyY(dominant_wavelength + bb_shift * amount);
    float3 adjutsed_xyY =
        float3(white_D65_xy + (shifted_xyY.xy - white_D65_xy) * white_offset_magnitude / max(1e-10, length((shifted_xyY.xy - white_D65_xy))), xyY.z);
    return CIE_xyY_to_XYZ(adjutsed_xyY);
}

#ifdef DECLARE_BEZOLD_BRUCKE_LUT
    DECLARE_BEZOLD_BRUCKE_LUT;

    // Apply Bezold–Brucke shift to XYZ stimulus. Loosely based on
    // "Pridmore, R. W. (1999). Bezold–Brucke hue-shift as functions of luminance level,
    // luminance ratio, interstimulus interval and adapting white for aperture and object colors.
    // Vision Research, 39(23), 3873–3891. doi:10.1016/s0042-6989(99)00085-1"
    float3 bezold_brucke_shift_XYZ_with_lut(float3 XYZ, float amount) {
        const float2 white = white_D65_xy;

        const float3 xyY = CIE_XYZ_to_xyY(XYZ);
        const float2 offset = xyY.xy - white;

        const float lut_coord = bb_xy_white_offset_to_lut_coord(offset);

        const float2 shifted_xy = xyY.xy + SAMPLE_BEZOLD_BRUCKE_LUT(lut_coord) * length(offset) * amount;
        return CIE_xyY_to_XYZ(float3(shifted_xy, xyY.z));
    }
#endif  // DECLARE_BEZOLD_BRUCKE_LUT
