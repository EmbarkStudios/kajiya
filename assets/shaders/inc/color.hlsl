#ifndef COLOR_HLSL
#define COLOR_HLSL

float3 hsv_to_rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float3 uint_id_to_color(uint id) {
    return float3(id % 11, id % 29, id % 7) / float3(10, 28, 6);
}

float linear_to_srgb(float v) {
    if (v <= 0.0031308) {
        return v * 12.92;
    } else {
        return pow(v, (1.0/2.4)) * (1.055) - 0.055;
    }
}

float3 linear_to_srgb(float3 v) {
    return float3(
        linear_to_srgb(v.x), 
        linear_to_srgb(v.y), 
        linear_to_srgb(v.z));
}

float srgb_to_linear(float v) {
    if (v <= 0.04045)
        return v / 12.92;
    else
        return pow((v + 0.055) / 1.055, 2.4);
}

float3 srgb_to_linear(float3 v) {
    return float3(
        srgb_to_linear(v.x),
        srgb_to_linear(v.y),
        srgb_to_linear(v.z));
}

float3 rgb_to_ycbcr(float3 col) {
    float3x3 m = float3x3(0.2126, 0.7152, 0.0722, -0.1146,-0.3854, 0.5, 0.5,-0.4542,-0.0458);
    return mul(m, col);
}

float3 ycbcr_to_rgb(float3 col) {
    float3x3 m = float3x3(1.0, 0.0, 1.5748, 1.0, -0.1873, -.4681, 1.0, 1.8556, 0.0);
    return max(0.0, mul(m, col));
}

// Rec. 709
float calculate_luma(float3 col) {
    return dot(float3(0.2126, 0.7152, 0.0722), col);
}

float3 linear_srgb_to_oklab(float3 c) {
    float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
    float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
    float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;

    float l_ = pow(l, 1.0 / 3.0);
    float m_ = pow(m, 1.0 / 3.0);
    float s_ = pow(s, 1.0 / 3.0);

    return float3(
        0.2104542553f*l_ + 0.7936177850f*m_ - 0.0040720468f*s_,
        1.9779984951f*l_ - 2.4285922050f*m_ + 0.4505937099f*s_,
        0.0259040371f*l_ + 0.7827717662f*m_ - 0.8086757660f*s_
    );
}

float3 oklab_to_linear_srgb(float3 c) {
    float l_ = c.x + 0.3963377774f * c.y + 0.2158037573f * c.z;
    float m_ = c.x - 0.1055613458f * c.y - 0.0638541728f * c.z;
    float s_ = c.x - 0.0894841775f * c.y - 1.2914855480f * c.z;

    float l = l_*l_*l_;
    float m = m_*m_*m_;
    float s = s_*s_*s_;

    return float3(
        +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
        -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
        -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s
    );
}

// https://www.shadertoy.com/view/4tdGWM
// T: absolute temperature (K)
float3 blackbody_radiation(float T) {
    float3 O = 0.0.xxx;
    
/*  // --- with physical units: (but math conditionning can be an issue)
    float h = 6.6e-34, k=1.4e-23, c=3e8; // Planck, Boltzmann, light speed  constants

    for (float i=0.; i<3.; i++) {  // +=.1 if you want to better sample the spectrum.
        float f = 4e14 * (1.+.5*i); 
        O[int(i)] += 1e7/m* 2.*(h*f*f*f)/(c*c) / (exp((h*f)/(k*T)) - 1.);  // Planck law
    }
*/
    // --- with normalized units:  f = 1 (red) to 2 (violet). 
    // const 19E3 also disappears if you normalized temperatures with 1 = 19000 K
     for (float i=0.; i<3.; i++) {  // +=.1 if you want to better sample the spectrum.
        float f = 1.+.5*i; 
        O[int(i)] += 10. * (f*f*f) / (exp((19E3*f/T)) - 1.);  // Planck law
    }

    return O;
}

#endif
