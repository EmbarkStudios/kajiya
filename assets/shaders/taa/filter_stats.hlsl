[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] RWTexture2D<float> output_tex;

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    float input_prob = input_tex[px];

    #if 1
        const int k = 1;
        {for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float prob = input_tex[px + int2(x, y)].x;
                input_prob = max(input_prob, prob);
            }
        }}
    #else
        const int k = 1;

        float ex = 0;
        float wsum = 0;
        {for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float prob = input_tex[px + int2(x, y)].x;
                float w = 1;
                ex += prob * w;
                wsum += w;
            }
        }}
        ex /= wsum;
        float threshold = ex;

        ex = 0;
        wsum = 0;
        {for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                float prob = input_tex[px + int2(x, y)].x;
                float w = prob > threshold ? 1 : 0.001;
                ex += prob * w;
                wsum += w;
            }
        }}
        ex /= wsum;
        input_prob = ex;
    #endif

    output_tex[px] = input_prob;
}
