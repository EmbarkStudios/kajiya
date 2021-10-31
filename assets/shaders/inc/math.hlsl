#ifndef MATH_HLSL
#define MATH_HLSL

#include "math_const.hlsl"

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
    return abs(primary_depth / max(1e-20, secondary_depth) - 1.0);
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
};

#endif
