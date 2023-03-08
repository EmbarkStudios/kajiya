#ifndef NOTORIOUS6_SRGB_HLSL
#define NOTORIOUS6_SRGB_HLSL

float sRGB_to_luminance(float3 col) {
    return dot(col, float3(0.2126, 0.7152, 0.0722));
}

// from Alex Tardiff: http://alextardif.com/Lightness.html
// Convert RGB with sRGB/Rec.709 primaries to CIE XYZ
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
float3 sRGB_to_XYZ(float3 color) {
    return mul(float3x3(
        0.4124564,  0.3575761,  0.1804375,
        0.2126729,  0.7151522,  0.0721750,
        0.0193339,  0.1191920,  0.9503041
    ), color);
}

// from Alex Tardiff: http://alextardif.com/Lightness.html
// Convert CIE XYZ to RGB with sRGB/Rec.709 primaries
// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
float3 XYZ_to_sRGB(float3 color) {
    return mul(float3x3(
         3.2404542, -1.5371385, -0.4985314,
        -0.9692660,  1.8760108,  0.0415560,
         0.0556434, -0.2040259,  1.0572252
    ), color);
}

float sRGB_OETF(float a) {
	return select(.0031308f >= a, 12.92f * a, 1.055f * pow(a, .4166666666666667f) - .055f);
}

float3 sRGB_OETF(float3 a) {
	return float3(sRGB_OETF(a.r), sRGB_OETF(a.g), sRGB_OETF(a.b));
}

float sRGB_EOTF(float a) {
	return select(.04045f < a, pow((a + .055f) / 1.055f, 2.4f), a / 12.92f);
}

float3 sRGB_EOTF(float3 a) {
	return float3(sRGB_EOTF(a.r), sRGB_EOTF(a.g), sRGB_EOTF(a.b));
}

#endif  // NOTORIOUS6_SRGB_HLSL
