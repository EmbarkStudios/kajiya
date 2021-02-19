#include "math_const.hlsl"

// Based on https://github.com/hdachev/glsl-atmosphere

#define iSteps 16
#define jSteps 8

#define iStepGrowth 1.4
#define jStepGrowth 1.4

// Frostbite, see the `a` in:
// https://youtu.be/zs0oYjwjNEo?t=5063
// something related to Bruneton's 2008 paper.
// not sure i'm applying this properly but whatever.
#define mieExtinctionMul 1.11 // 1.11 in frostbite

// why the fuck do i need to crank this shit
#define ozoMul 6.00

float2 rsi(float3 r0, float3 rd, float sr) {
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, r0);
    float c = dot(r0, r0) - (sr * sr);
    float d = (b*b) - 4.0*a*c;
    if (d < 0.0) return float2(1e5,-1e5);
    return float2(
        (-b - sqrt(d))/(2.0*a),
        (-b + sqrt(d))/(2.0*a)
    );
}

float geometricSeries(float commonRatio, float numTerms) {
    // Sum of first n terms in a geometric progression starting with 1:
    // a * ( 1 - r^n ) / ( 1 - r ), here a is 1.
    return (1.0 - pow(commonRatio, numTerms))
         / (1.0 - commonRatio);
}

float3 atmosphere1(
    float3 r, float3 r0, float3 pSun,
    float rPlanet, float rAtmos,
    float3 kRlh, float kMie, float shRlh, float shMie,
    float g) {

    // Normalize the sun and view directions.
    pSun = normalize(pSun);
    r = normalize(r);

    // Calculate the step size of the primary ray.
    float2 p = rsi(r0, r, rAtmos);
    if (p.x > p.y) return float3(0,0,0);
    p.y = min(p.y, rsi(r0, r, rPlanet).x);

    float iDist = p.y - p.x;
    float iStepSize = iDist
                    / geometricSeries(iStepGrowth, float(iSteps));

    // Initialize the primary ray time.
    float iTime = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    float3 totalRlh = float3(0,0,0);
    float3 totalMie = float3(0,0,0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    float mu = dot(r, pSun);
    float mumu = mu * mu;
    float gg = g * g;

    float phaseRlh = 3.0 / (16.0 * M_PI) * (1.0 + mumu);
    float phaseMie = 3.0 / (8.0 * M_PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {

if (i != -1) {

        // Calculate the primary ray sample position.
        float3 iPos = r0 + r * (iTime + iStepSize * 0.5);

        // Calculate the height of the sample.
        float iHeight = length(iPos) - rPlanet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        float odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        // Calculate the step size of the secondary ray.
        float jStepSize = rsi(iPos, pSun, rAtmos).y
                        / geometricSeries(jStepGrowth, float(jSteps));

        // Initialize the secondary ray time.
        float jTime = 0.0;

        // Initialize optical depth accumulators for the secondary ray.
        float jOdRlh = 0.0;
        float jOdMie = 0.0;

        // Sample the secondary ray.
        for (int j = 0; j < jSteps; j++) {

            // Calculate the secondary ray sample position.
            float3 jPos = iPos + pSun * (jTime + jStepSize * 0.5);

            // Calculate the height of the sample.
            float jHeight = length(jPos) - rPlanet;

            // Accumulate the optical depth.
            jOdRlh += exp(-jHeight / shRlh) * jStepSize;
            jOdMie += exp(-jHeight / shMie) * jStepSize;

            // Increment the secondary ray time.
            jTime += jStepSize;
            jStepSize *= jStepGrowth;
        }

        // Calculate attenuation.
        float3 attn = exp(
            -(   kMie * (iOdMie + jOdMie)
               + kRlh * (iOdRlh + jOdRlh)));

        // Accumulate scattering.
        totalRlh += odStepRlh * attn;
        totalMie += odStepMie * attn;
}
        // Increment the primary ray time.
        iTime += iStepSize;
        iStepSize *= iStepGrowth;
    }

    // Calculate and return the final color.
    return phaseRlh * kRlh * totalRlh
         + phaseMie * kMie * totalMie;
}


float3 atmosphere2(
    float3 r, float3 r0, float3 pSun,
    float rPlanet, float rAtmos,
    float3 kRlh, float kMie, float3 kOzo,
    float shRlh, float shMie,
    float g) {

    // Normalize the sun and view directions.
    pSun = normalize(pSun);
    r = normalize(r);

    // Calculate the step size of the primary ray.
    float2 p = rsi(r0, r, rAtmos);
    if (p.x > p.y) return float3(0,0,0);
    p.y = min(p.y, rsi(r0, r, rPlanet).x);
    float iDist = p.y - p.x;
    float iStepSize = iDist
                    / geometricSeries(iStepGrowth, float(iSteps));

    // Initialize the primary ray time.
    float iTime = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    float3 totalRlh = float3(0,0,0);
    float3 totalMie = float3(0,0,0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    float mu = dot(r, pSun);
    float mumu = mu * mu;
    float gg = g * g;

    float phaseRlh = 3.0 / (16.0 * M_PI) * (1.0 + mumu);
    float phaseMie = 3.0 / (8.0 * M_PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {

if (i != -1) {

        // Calculate the primary ray sample position.
        float3 iPos = r0 + r * (iTime + iStepSize * 0.5);

        // Calculate the height of the sample.
        float iHeight = length(iPos) - rPlanet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        float odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        // Calculate the step size of the secondary ray.
        float jStepSize = rsi(iPos, pSun, rAtmos).y
                        / geometricSeries(jStepGrowth, float(jSteps));

        // Initialize the secondary ray time.
        float jTime = 0.0;

        // Initialize optical depth accumulators for the secondary ray.
        float jOdRlh = 0.0;
        float jOdMie = 0.0;

        // Sample the secondary ray.
        for (int j = 0; j < jSteps; j++) {

            // Calculate the secondary ray sample position.
            float3 jPos = iPos + pSun * (jTime + jStepSize * 0.5);

            // Calculate the height of the sample.
            float jHeight = length(jPos) - rPlanet;

            // Accumulate the optical depth.
            jOdRlh += exp(-jHeight / shRlh) * jStepSize;
            jOdMie += exp(-jHeight / shMie) * jStepSize;

            // Increment the secondary ray time.
            jTime += jStepSize;
            jStepSize *= jStepGrowth;
        }

        // Calculate attenuation.
        float3 attn = exp(
            -(   kMie * (iOdMie + jOdMie) * mieExtinctionMul
               + kRlh * (iOdRlh + jOdRlh)
               + kOzo * (iOdRlh + jOdRlh) * ozoMul));

        // Accumulate scattering.
        totalRlh += odStepRlh * attn;
        totalMie += odStepMie * attn;
}
        // Increment the primary ray time.
        iTime += iStepSize;
        iStepSize *= iStepGrowth;
    }

    // Calculate and return the final color.
    return phaseRlh * kRlh * totalRlh
         + phaseMie * kMie * totalMie;
}

float3 atmosphere_default(float3 wi, float3 to_light) {
    //return wi * 0.5 + 0.5;
    //return 1.0.xxx;

    return atmosphere2(
        wi,                 // normalized ray direction
        float3(0,6371e3,0),               // ray origin
        // sun pos
        to_light,
        6371e3,                         // radius of the planet in meters
        6471e3,                         // radius of the atmosphere in meters
        float3(5.8e-6, 13.5e-6, 33.1e-6), // frostbite
        21e-6,                          // Mie scattering coefficient
        float3(3.426e-7, 8.298e-7, 0.356e-7), // Ozone extinction, frostbite
        8e3,                            // Rayleigh scale height
        1.2e3,                          // Mie scale height
        0.758                           // Mie preferred scattering direction
    ) * 20.0;
}