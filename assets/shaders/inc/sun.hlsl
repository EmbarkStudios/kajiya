#ifndef SUN_HLSL
#define SUN_HLSL

#include "frame_constants.hlsl"

// static const float3 SUN_DIRECTION = normalize(float3(1, 1.6, -0.2));
// static const float3 SUN_DIRECTION = normalize(float3(-0.8, 0.3, 1.0));

#define SUN_DIRECTION (frame_constants.sun_direction.xyz)

static const float3 SUN_COLOR = float3(1.6, 1.2, 0.9) * 5.0 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);
//static const float3 SUN_COLOR = 5 * atmosphere_default(SUN_DIRECTION, SUN_DIRECTION);

#endif