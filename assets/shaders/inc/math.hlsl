#ifndef MATH_HLSL
#define MATH_HLSL

#include "math_const.hlsl"

float max3(float x, float y, float z) {
    return max(x, max(y, z));
}

float square(float x) { return x * x; }
float2 square(float2 x) { return x * x; }
float3 square(float3 x) { return x * x; }
float4 square(float4 x) { return x * x; }

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
float3x3 build_orthonormal_basis(float3 n) {
    float3 b1;
    float3 b2;

    if (n.z < 0.0) {
        const float a = 1.0 / (1.0 - n.z);
        const float b = n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = float3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const float a = 1.0 / (1.0 + n.z);
        const float b = -n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return float3x3(
        b1.x, b2.x, n.x,
        b1.y, b2.y, n.y,
        b1.z, b2.z, n.z
    );
}

float3 uniform_sample_cone(float2 urand, float cos_theta_max) {
    float cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(saturate(1.0 - cos_theta * cos_theta));
    float phi = urand.y * M_TAU;
    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Calculates vector d such that
// lerp(a, d.rgb, d.a) equals lerp(lerp(a, b.rgb, b.a), c.rgb, c.a)
//
// Lerp[a_, b_, c_] := a  (1-c) + b  c
// FullSimplify[Lerp[a,(b(c (1 -  e)) + d e) /(c + e - c e), 1-(1-c)(1-e)]] == FullSimplify[Lerp[Lerp[a, b, c], d, e]]
float4 prelerp(float4 b, float4 c) {
    float denom = b.a + c.a * (1.0 - b.a);
    return denom > 1e-5 ? float4(
        (b.rgb * (b.a * (1.0 - c.a)) + c.rgb * c.a) / denom,
        1.0 - (1.0 - b.a) * (1.0 - c.a)
    ) : 0.0.xxxx;
}

float inverse_depth_relative_diff(float primary_depth, float secondary_depth) {
    return abs(max(1e-20, primary_depth) / max(1e-20, secondary_depth) - 1.0);
}

// Encode a scalar a space which heavily favors small values.
float exponential_squish(float len, float squish_strength) {
    return exp2(-clamp(squish_strength * len, 0, 100));
}

// Ditto, decode.
float exponential_unsquish(float len, float squish_strength) {
    return max(0.0, -1.0 / squish_strength * log2(1e-30 + len));
}

#endif
