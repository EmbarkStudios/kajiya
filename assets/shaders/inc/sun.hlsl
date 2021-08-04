#ifndef SUN_HLSL
#define SUN_HLSL

#include "frame_constants.hlsl"
#include "math.hlsl"

// static const float3 SUN_DIRECTION = normalize(float3(1, 1.6, -0.2));
// static const float3 SUN_DIRECTION = normalize(float3(-0.8, 0.3, 1.0));

#define SUN_DIRECTION (frame_constants.sun_direction.xyz)
//#define SUN_DIRECTION float3(0, -1, 0)
 //#define SUN_DIRECTION normalize(float3(-6.0, 0.5, -1.5))

//static const float3 SUN_COLOR = 5 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

#if 0
    static const float3 SUN_COLOR = 1.0;
#else
    #if USE_FELIX_ATMOSPHERE
        float3 sun_color_in_direction(float3 dir) {
            return
                20.0 *
                frame_constants.sun_color_multiplier.rgb *
                Absorb(IntegrateOpticalDepth(0.0.xxx, dir));
        }

        #define SUN_COLOR (sun_color_in_direction(SUN_DIRECTION))
        //#define SUN_COLOR 0.0
    #else
        static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);
    #endif
#endif

float3 sample_sun_direction(float2 urand, bool soft) {
    if (soft) {
        if (frame_constants.sun_angular_radius_cos < 1.0) {
            const float3x3 basis = build_orthonormal_basis(normalize(SUN_DIRECTION));
            return mul(basis, uniform_sample_cone(urand, frame_constants.sun_angular_radius_cos));
        }
    }

    return SUN_DIRECTION;
}

#endif