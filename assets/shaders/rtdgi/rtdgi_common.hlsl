#ifndef RTDGI_COMMON_HLSL
#define RTDGI_COMMON_HLSL

float4 decode_hit_normal_and_dot(float4 val) {
    return float4(val.xyz * 2 - 1, val.w);
}

float4 encode_hit_normal_and_dot(float4 val) {
    return float4(val.xyz * 0.5 + 0.5, val.w);
}

#endif  // RTDGI_COMMON_HLSL
