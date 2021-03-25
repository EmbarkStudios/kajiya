#ifndef ATMOSPHERE_HLSL
#define ATMOSPHERE_HLSL

#include "atmosphere_felix.hlsl"

#define USE_FELIX_ATMOSPHERE 1

float3 atmosphere_default(float3 wi, float3 light_dir) {
    // return 0.5;

    float3 _WorldSpaceCameraPos = float3(0, 0, 0);
    float3 rayStart  = _WorldSpaceCameraPos;
    float3 rayDir    = wi;
    float  rayLength = INFINITY;

    float3 lightDir   = light_dir;
    float3 lightColor = 1.0.xxx;

    float3 transmittance;
    return IntegrateScattering(rayStart, rayDir, rayLength, lightDir, lightColor, transmittance);
}

#endif