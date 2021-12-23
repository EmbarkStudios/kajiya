#include "../inc/math.hlsl"

#define TAA_NONLINEARITY_TYPE 1
#define TAA_COLOR_MAPPING_MODE 1


float linear_to_perceptual(float a) {
    #if 0 == TAA_NONLINEARITY_TYPE
        return a;
    #elif 1 == TAA_NONLINEARITY_TYPE
        return sqrt(max(0.0, a));
    #elif 2 == TAA_NONLINEARITY_TYPE
        return max(0.0, log(1.0 + sqrt(max(0.0, a))));
    #elif 3 == TAA_NONLINEARITY_TYPE
        return max(0.0, 1.0 - exp(-max(0.0, a)));
    #elif 4 == TAA_NONLINEARITY_TYPE
        const float k = 0.25;   // Linear part end

        return a < k
            ? max(0.0, a)
            : k - 0.5 + sqrt(a - k + 0.25);
    #else
        return 0;
    #endif
}

float perceptual_to_linear(float a) {
    #if 0 == TAA_NONLINEARITY_TYPE
        return a;
    #elif 1 == TAA_NONLINEARITY_TYPE
        return a * a;
    #elif 2 == TAA_NONLINEARITY_TYPE
        a = exp(a) - 1.0;
        return a * a;
    #elif 3 == TAA_NONLINEARITY_TYPE
        return max(0.0, -log(1.0 - a));
    #elif 4 == TAA_NONLINEARITY_TYPE
        const float k = 0.25;   // Linear part end

        return a < k
            ? max(0.0, a)
            : square(a - k + 0.5) + k - 0.25;
    #else
        return 0;
    #endif
}

float3 decode_rgb(float3 v) {
    #if 0 == TAA_COLOR_MAPPING_MODE
        return float3(linear_to_perceptual(v.r), linear_to_perceptual(v.g), linear_to_perceptual(v.b));
    #elif 1 == TAA_COLOR_MAPPING_MODE
        float max_comp = max3(v.r, v.g, v.b);
        return v * linear_to_perceptual(max_comp) / max(1e-20, max_comp);
    #endif
}

float3 encode_rgb(float3 v) {
    #if 0 == TAA_COLOR_MAPPING_MODE
        return float3(perceptual_to_linear(v.r), perceptual_to_linear(v.g), perceptual_to_linear(v.b));
    #elif 1 == TAA_COLOR_MAPPING_MODE
        float max_comp = max3(v.r, v.g, v.b);
        return v * perceptual_to_linear(max_comp) / max(1e-20, max_comp);
    #endif
}
