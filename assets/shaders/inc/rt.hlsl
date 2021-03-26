#ifndef RT_HLSL
#define RT_HLSL

#include "math_const.hlsl"
#include "gbuffer.hlsl"

struct GbufferRayPayload {
    GbufferDataPacked gbuffer_packed;
    float t;
    float cone_width;

    static GbufferRayPayload new_miss() {
        GbufferRayPayload res;
        res.t = FLT_MAX;
        res.cone_width = 0;
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

    static ShadowRayPayload new_hit() {
        ShadowRayPayload res;
        res.is_shadowed = true;
        return res;
    }

    bool is_miss() {
        return !is_shadowed;
    }

    bool is_hit() {
        return !is_miss();
    }
};

RayDesc new_ray(float3 origin, float3 direction, float tmin, float tmax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tmin;
    ray.TMax = tmax;
    return ray;
}

bool rt_is_shadowed(
    RaytracingAccelerationStructure acceleration_structure,
    RayDesc ray
) {
    ShadowRayPayload shadow_payload = ShadowRayPayload::new_hit();
    TraceRay(
        acceleration_structure,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xff, 0, 0, 1, ray, shadow_payload
    );

    return shadow_payload.is_shadowed;
}

struct GbufferPathVertex {
    bool is_hit;
    GbufferDataPacked gbuffer_packed;
    float3 position;
    float ray_t;
};

GbufferPathVertex rt_trace_gbuffer(
    RaytracingAccelerationStructure acceleration_structure,
    RayDesc ray,
    float cone_width
) {
    GbufferRayPayload payload = GbufferRayPayload::new_miss();
    payload.cone_width = cone_width;

    TraceRay(acceleration_structure, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xff, 0, 0, 0, ray, payload);

    if (payload.is_hit()) {
        GbufferPathVertex res;
        res.is_hit = true;
        res.position = ray.Origin + ray.Direction * payload.t;
        res.gbuffer_packed = payload.gbuffer_packed;
        res.ray_t = payload.t;
        return res;
    } else {
        GbufferPathVertex res;
        res.is_hit = false;
        res.ray_t = FLT_MAX;
        return res;
    }
}

GbufferPathVertex rt_trace_gbuffer_nocull(
    RaytracingAccelerationStructure acceleration_structure,
    RayDesc ray,
    float cone_width
) {
    GbufferRayPayload payload = GbufferRayPayload::new_miss();
    payload.cone_width = cone_width;
    
    TraceRay(acceleration_structure, 0, 0xff, 0, 0, 0, ray, payload);

    if (payload.is_hit()) {
        GbufferPathVertex res;
        res.is_hit = true;
        res.position = ray.Origin + ray.Direction * payload.t;
        res.gbuffer_packed = payload.gbuffer_packed;
        res.ray_t = payload.t;
        return res;
    } else {
        GbufferPathVertex res;
        res.is_hit = false;
        res.ray_t = FLT_MAX;
        return res;
    }
}

#endif