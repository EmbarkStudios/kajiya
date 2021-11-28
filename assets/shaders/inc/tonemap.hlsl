#include "color.hlsl"

float tonemap_curve(float v) {
    #if 0
        // Large linear part in the lows, but compresses highs.
        float c = v + v*v + 0.5*v*v*v;
        return c / (1.0 + c);
    #else
        return 1.0 - exp(-v);
    #endif
}

float3 tonemap_curve(float3 v) {
    return float3(tonemap_curve(v.r), tonemap_curve(v.g), tonemap_curve(v.b));
}

float3 neutral_tonemap(float3 col) {
    float3 ycbcr = rgb_to_ycbcr(col);

    float bt = tonemap_curve(length(ycbcr.yz) * 2.4);
    float desat = max((bt - 0.7) * 0.8, 0.0);
    desat *= desat;

    float3 desat_col = lerp(col.rgb, ycbcr.xxx, desat);

    float tm_luma = tonemap_curve(ycbcr.x);
    float3 tm0 = col.rgb * max(0.0, tm_luma / max(1e-5, calculate_luma(col.rgb)));
    float final_mult = 0.97;
    float3 tm1 = tonemap_curve(desat_col);

    col = lerp(tm0, tm1, bt * bt);

    return col * final_mult;
}
