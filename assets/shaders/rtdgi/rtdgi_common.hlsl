#ifndef RTDGI_COMMON_HLSL
#define RTDGI_COMMON_HLSL

float4 decode_hit_normal_and_dot(float4 val) {
    return float4(val.xyz * 2 - 1, val.w);
}

float4 encode_hit_normal_and_dot(float4 val) {
    return float4(val.xyz * 0.5 + 0.5, val.w);
}

struct TemporalReservoirOutput {
    float depth;
    float3 ray_hit_offset_ws;
    float luminance;
    float3 hit_normal_ws;

    static TemporalReservoirOutput from_raw(uint4 raw) {
        float4 ray_hit_offset_and_luminance = float4(
            unpack_2x16f_uint(raw.y),
            unpack_2x16f_uint(raw.z));

        TemporalReservoirOutput res;
        res.depth = asfloat(raw.x);
        res.ray_hit_offset_ws = ray_hit_offset_and_luminance.xyz;
        res.luminance = ray_hit_offset_and_luminance.w;
        res.hit_normal_ws = unpack_normal_11_10_11(asfloat(raw.w));
        return res;
    }

    uint4 as_raw() {
        uint4 raw;
        raw.x = asuint(depth);
        raw.y = pack_2x16f_uint(ray_hit_offset_ws.xy);
        raw.z = pack_2x16f_uint(float2(ray_hit_offset_ws.z, luminance));
        raw.w = asuint(pack_normal_11_10_11(hit_normal_ws));
        return raw;
    }
};

#endif  // RTDGI_COMMON_HLSL
