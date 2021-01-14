#ifndef RT_HLSL
#define RT_HLSL

#include "math_const.hlsl"

struct GbufferRayPayload {
    float4 gbuffer_packed;
    float t;

    static GbufferRayPayload new_miss() {
        GbufferRayPayload res;
        res.gbuffer_packed = 0.0.xxxx;
        res.t = FLT_MAX;
        return res;
    }

    bool is_miss() {
        return t == FLT_MAX;
    }

    bool is_hit() {
        return !is_miss();
    }
};

struct ShadowRayPayload {
    bool is_shadowed;

    static ShadowRayPayload new_miss() {
        ShadowRayPayload res;
        res.is_shadowed = false;
        return res;
    }

    bool is_miss() {
        return !is_shadowed;
    }

    bool is_hit() {
        return !is_miss();
    }
};

#endif