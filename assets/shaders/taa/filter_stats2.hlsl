[[vk::binding(0)]] Texture2D<float2> input_tex;
[[vk::binding(1)]] RWTexture2D<float2> output_tex;

// Encode ray length in a space which heavily favors short ones.
// For temporal averaging of distance to ray hits.
float squish_ray_len(float len, float squish_strength) {
    return exp2(-clamp(squish_strength * len, 0, 100));
}

// Ditto, decode.
float unsquish_ray_len(float len, float squish_strength) {
    return max(0.0, -1.0 / squish_strength * log2(1e-30 + len));
}

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    float2 input_stats = input_tex[px];

    float2 derp = 0;

    const int k = 2;
    const float m = 10;

    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float2 stats = input_tex[px + int2(x, y) * 2];
            input_stats.x = min(input_stats.x, stats.x);
            derp += float2(squish_ray_len(stats.x, m), 1);
        }
    }

    input_stats.x = unsquish_ray_len(derp.x / derp.y, m);

    //input_stats.x = min(input_stats.x, WaveReadLaneAt(input_stats.x, WaveGetLaneIndex() ^ 1));
    //input_stats.x = min(input_stats.x, WaveReadLaneAt(input_stats.x, WaveGetLaneIndex() ^ 8));

    output_tex[px] = input_stats;
}
