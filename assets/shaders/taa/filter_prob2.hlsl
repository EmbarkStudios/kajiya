#include "../inc/math.hlsl"

[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] RWTexture2D<float> output_tex;

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    float prob = input_tex[px];

    float2 weighted_prob = 0;
    const float SQUISH_STRENGTH = 10;

    const int k = 2;
    for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            float neighbor_prob = input_tex[px + int2(x, y) * 2];
            weighted_prob += float2(exponential_squish(neighbor_prob, SQUISH_STRENGTH), 1);
        }
    }

    prob = exponential_unsquish(weighted_prob.x / weighted_prob.y, SQUISH_STRENGTH);

    //prob = min(prob, WaveReadLaneAt(prob, WaveGetLaneIndex() ^ 1));
    //prob = min(prob, WaveReadLaneAt(prob, WaveGetLaneIndex() ^ 8));

    output_tex[px] = prob;
}
