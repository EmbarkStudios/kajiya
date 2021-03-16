#include "color.hlsl"

float tonemap_curve(float v) {
    float c = v + v*v + 0.5*v*v*v;
    return c / (1.0 + c);
}

float tonemap_curve(float v, float lin_part) {
    #if 0
        float c = v + v*v + 0.5*v*v*v;
        return c / (1.0 + c);
    #elif 0
        return 1 - exp(-v);
    #else
        float n = 1.0 + exp(3 * lin_part);
        return 1 - (1.0 - v) / (1.0 - pow(v, n));
    #endif
}

float3 tonemap_curve(float3 v, float lin_part) {
    return float3(tonemap_curve(v.r, lin_part), tonemap_curve(v.g, lin_part), tonemap_curve(v.b, lin_part));
}

float3 neutral_tonemap(float3 col, float lin_part) {
    //float3x3 ycbr_mat_t = float3x3(.2126, .7152, .0722, -.1146,-.3854, .5, .5,-.4542,-.0458);
    //float3 ycbcr = mul(ycbr_mat_t, col);
    float3 ycbcr = rgb_to_ycbcr(col);

    float bt = tonemap_curve(length(ycbcr.yz) * 2.4, lin_part);
    float desat = max((bt - 0.7) * 0.8, 0.0);
    desat *= desat;

    float3 desat_col = lerp(col.rgb, ycbcr.xxx, desat);

    float tm_luma = tonemap_curve(ycbcr.x, lin_part);
    float3 tm0 = col.rgb * max(0.0, tm_luma / max(1e-5, calculate_luma(col.rgb)));
    float final_mult = 0.97;
    float3 tm1 = tonemap_curve(desat_col, lin_part);

    col = lerp(tm0, tm1, bt * bt);

    return col * final_mult;
}
