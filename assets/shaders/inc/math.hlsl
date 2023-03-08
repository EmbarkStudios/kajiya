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

float length_squared(float2 v) { return dot(v, v); }
float length_squared(float3 v) { return dot(v, v); }
float length_squared(float4 v) { return dot(v, v); }

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
    return select(denom > 1e-5, float4(
        (b.rgb * (b.a * (1.0 - c.a)) + c.rgb * c.a) / denom,
        1.0 - (1.0 - b.a) * (1.0 - c.a)
    ), 0.0.xxxx);
}

float inverse_depth_relative_diff(float primary_depth, float secondary_depth) {
    return abs(max(1e-20, primary_depth) / max(1e-20, secondary_depth) - 1.0);
}

// Encode a scalar a space which heavily favors small values.
float exponential_squish(float len, float squish_scale) {
    return exp2(-clamp(squish_scale * len, 0, 100));
}

// Ditto, decode.
float exponential_unsquish(float len, float squish_scale) {
    return max(0.0, -1.0 / squish_scale * log2(1e-30 + len));
}

float3 uniform_sample_hemisphere(float2 urand) {
     float phi = urand.y * M_TAU;
     float cos_theta = 1.0 - urand.x;
     float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
     return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

float3 uniform_sample_sphere(float2 urand) {
    float z = 1.0 - 2.0 * urand.x;
    float xy = sqrt(max(0.0, 1.0 - z * z));
    float sn = sin(M_TAU * urand.y);
	float cs = cos(M_TAU * urand.y);
	return float3(cs * xy, sn * xy, z);
}

struct RaySphereIntersection {
    // `t` will be NaN if no intersection is found
    float t;
    float3 normal;

    bool is_hit() {
        return t >= 0.0;
    }
};

struct Sphere {
    float3 center;
    float radius;

    static Sphere from_center_radius(float3 center, float radius) {
        Sphere res;
        res.center = center;
        res.radius = radius;
        return res;
    }

    RaySphereIntersection intersect_ray(float3 ray_origin, float3 ray_dir) {
    	float3 oc = ray_origin - center;
        float a = dot(ray_dir, ray_dir);
    	float b = 2.0 * dot(oc, ray_dir);
    	float c = dot(oc, oc) - radius * radius;
    	float h = b * b - 4.0 * a * c;

        RaySphereIntersection res;
        res.t = (-b - sqrt(h)) / (2.0 * a);
        res.normal = normalize(ray_origin + res.t * ray_dir - center);
        return res;
    }

    RaySphereIntersection intersect_ray_inside(float3 ray_origin, float3 ray_dir) {
    	float3 oc = ray_origin - center;
        float a = dot(ray_dir, ray_dir);
    	float b = 2.0 * dot(oc, ray_dir);
    	float c = dot(oc, oc) - radius * radius;

    	float h = b * b - 4.0 * a * c;

        RaySphereIntersection res;
        // Note: flipped sign compared to the regular version
        res.t = (-b + sqrt(h)) / (2.0 * a);
        res.normal = normalize(ray_origin + res.t * ray_dir - center);
        return res;
    }
};

struct RayBoxIntersection {
    float t_min;
    float t_max;

    bool is_hit() {
        return t_min <= t_max && t_max >= 0.0;
    }
};

struct Aabb {
    float3 pmin;
    float3 pmax;

    static Aabb from_min_max(float3 pmin, float3 pmax) {
        Aabb res;
        res.pmin = pmin;
        res.pmax = pmax;
        return res;
    }

    static Aabb from_center_half_extent(float3 center, float3 half_extent) {
        Aabb res;
        res.pmin = center - half_extent;
        res.pmax = center + half_extent;
        return res;
    }

    // From https://github.com/tigrazone/glslppm
    RayBoxIntersection intersect_ray(float3 ray_origin, float3 ray_dir) {
    	float3 min_interval = (pmax.xyz - ray_origin.xyz) / ray_dir;
    	float3 max_interval = (pmin.xyz - ray_origin.xyz) / ray_dir;

    	float3 a = min(min_interval, max_interval);
    	float3 b = max(min_interval, max_interval);

        RayBoxIntersection res;
        res.t_min = max(max(a.x, a.y), a.z);
        res.t_max = min(min(b.x, b.y), b.z);
        // return t_min <= t_max && t_min < t && t_max >= 0.0;
        return res;
    }

    bool contains_point(float3 p) {
        return all(p >= pmin) && all(p <= pmax);
    }
};

float inverse_lerp(float minv, float maxv, float v) {
    return (v - minv) / (maxv - minv);
}

#endif
