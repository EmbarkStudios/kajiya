#ifndef NOTORIOUS6_LUV_HLSL
#define NOTORIOUS6_LUV_HLSL

float luminance_to_LUV_L(float Y){
    return select(Y <= 0.0088564516790356308, Y * 903.2962962962963, 116.0 * pow(max(0.0, Y), 1.0 / 3.0) - 16.0);
}

float LUV_L_to_luminance(float L) {
    return select(L <= 8.0, L / 903.2962962962963, pow((L + 16.0) / 116.0, 3.0));
}

float2 CIE_xyY_xy_to_LUV_uv(float2 xy) {
    return xy * float2(4.0, 9.0) / (-2.0 * xy.x + 12.0 * xy.y + 3.0);
}

float2 CIE_XYZ_to_LUV_uv(float3 xyz) {
    return xyz.xy * float2(4.0, 9.0) / dot(xyz, float3(1.0, 15.0, 3.0));
}


#endif  // NOTORIOUS6_LUV_HLSL
