#define ENCODING_VARIANT 1

float linear_to_perceptual(float a) {
    #if 0 == ENCODING_VARIANT
        return a;
    #elif 1 == ENCODING_VARIANT
        return sqrt(max(0.0, a));
    #elif 2 == ENCODING_VARIANT
        return max(0.0, log(1.0 + sqrt(max(0.0, a))));
    #endif
}

float perceptual_to_linear(float a) {
    #if 0 == ENCODING_VARIANT
        return a;
    #elif 1 == ENCODING_VARIANT
        return a * a;
    #elif 2 == ENCODING_VARIANT
        a = exp(a) - 1.0;
        return a * a;
    #endif
}

float3 decode_rgb(float3 a) {
    return float3(linear_to_perceptual(a.r), linear_to_perceptual(a.g), linear_to_perceptual(a.b));
}

float3 encode_rgb(float3 a) {
    return float3(perceptual_to_linear(a.r), perceptual_to_linear(a.g), perceptual_to_linear(a.b));
}
