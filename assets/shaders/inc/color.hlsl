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

// Rec. 709
float calculate_luma(float3 col) {
	return dot(float3(0.2126, 0.7152, 0.0722), col);
}

#endif
