[[vk::binding(0)]] Texture2D<float2> input_tex;
[[vk::binding(1)]] RWTexture2D<float2> output_tex;

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    float2 input_stats = input_tex[px];

    #if 1
        const int k = 1;
        {for (int y = -k; y <= k; ++y) {
            for (int x = -k; x <= k; ++x) {
                if (abs(x) + abs(y) > 1) {
                    //continue;
                }

                float2 stats = input_tex[px + int2(x, y)];
                input_stats.x = max(input_stats.x, stats.x);
                //input_stats.y = max(input_stats.y, stats.y);
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
        input_stats = ex;
    #endif

    output_tex[px] = input_stats;
}
