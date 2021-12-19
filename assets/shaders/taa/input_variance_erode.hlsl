[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] RWTexture2D<float> output_tex;

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    float input_prob = input_tex[px];

    const int k = 1;
    {for (int y = -k; y <= k; ++y) {
        for (int x = -k; x <= k; ++x) {
            /*if (abs(x) + abs(y) > 1) {
                continue;
            }*/

            float prob = input_tex[px + int2(x, y)].x;
            input_prob = min(input_prob, prob);
        }
    }}

    output_tex[px] = input_prob;
}
