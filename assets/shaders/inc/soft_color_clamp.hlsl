float3 soft_color_clamp(float3 center, float3 history, float3 ex, float3 dev) {
    // Sort of like the color bbox clamp, but with a twist. In noisy surrounds, the bbox becomes
    // very large, and then the clamp does nothing, especially with a high multiplier on std. deviation.
    //
    // Instead of a hard clamp, this will smoothly bring the value closer to the center,
    // thus over time reducing disocclusion artifacts.
    float3 history_dist = abs(history - ex) / dev;
    //float3 closest_pt = center;
    float3 closest_pt = clamp(history, center - dev, center + dev);
    return lerp(history, closest_pt, smoothstep(1.0, 3.0, history_dist));
}
