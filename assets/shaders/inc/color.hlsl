#ifndef COLOR_HLSL
#define COLOR_HLSL

#include "color/srgb.hlsl"
#include "color/ycbcr.hlsl"

float3 uint_id_to_sRGB(uint id) {
    return float3(id % 11, id % 29, id % 7) / float3(10, 28, 6);
}

// https://www.shadertoy.com/view/4tdGWM
// T: absolute temperature (K)
float3 blackbody_radiation(float T) {
    float3 O = 0.0.xxx;
    
/*  // --- with physical units: (but math conditionning can be an issue)
    float h = 6.6e-34, k=1.4e-23, c=3e8; // Planck, Boltzmann, light speed  constants

    for (float i=0.; i<3.; i++) {  // +=.1 if you want to better sample the spectrum.
        float f = 4e14 * (1.+.5*i); 
        O[int(i)] += 1e7/m* 2.*(h*f*f*f)/(c*c) / (exp((h*f)/(k*T)) - 1.);  // Planck law
    }
*/
    // --- with normalized units:  f = 1 (red) to 2 (violet). 
    // const 19E3 also disappears if you normalized temperatures with 1 = 19000 K
     for (float i=0.; i<3.; i++) {  // +=.1 if you want to better sample the spectrum.
        float f = 1.+.5*i; 
        O[int(i)] += 10. * (f*f*f) / (exp((19E3*f/T)) - 1.);  // Planck law
    }

    return O;
}

#endif
