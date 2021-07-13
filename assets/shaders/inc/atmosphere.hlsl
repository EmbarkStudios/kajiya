#ifndef ATMOSPHERE_HLSL
#define ATMOSPHERE_HLSL

#include "atmosphere_felix.hlsl"
#include "frame_constants.hlsl"

#define USE_FELIX_ATMOSPHERE 1

float3 atmosphere_default(float3 wi, float3 light_dir) {
    //return max(0.0, normalize(wi) * 0.5 + 0.5);
    // return 0.5;

    float3 _WorldSpaceCameraPos = float3(0, 0, 0);
    float3 rayStart  = _WorldSpaceCameraPos;
    float3 rayDir    = wi;
    float  rayLength = INFINITY;

    float3 lightDir   = light_dir;
    float3 lightColor = 1.0.xxx;

    float3 transmittance;
    return
        frame_constants.sky_ambient.rgb +
        frame_constants.sun_color_multiplier.rgb *
            IntegrateScattering(rayStart, rayDir, rayLength, lightDir, lightColor, transmittance);
}

#endif