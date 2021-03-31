#ifndef CURVE_HLSL
#define CURVE_HLSL

float3 cubic_hermite(float3 A, float3 B, float3 C, float3 D, float t) {
	float t2 = t*t;
    float t3 = t*t*t;
    float3 a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0;
    float3 b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0;
    float3 c = -A/2.0 + C/2.0;
   	float3 d = B;
    
    return a*t3 + b*t2 + c*t + d;
}

float mitchell_netravali(float x) {
    float B = 1.0 / 3.0;
    float C = 1.0 / 3.0;

    float ax = abs(x);
    if (ax < 1) {
        return ((12 - 9 * B - 6 * C) * ax * ax * ax + (-18 + 12 * B + 6 * C) * ax * ax + (6 - 2 * B)) / 6;
    } else if ((ax >= 1) && (ax < 2)) {
        return ((-B - 6 * C) * ax * ax * ax + (6 * B + 30 * C) * ax * ax + (-12 * B - 48 * C) * ax + (8 * B + 24 * C)) / 6;
    } else {
        return 0;
    }
}

#endif  // CURVE_HLSL