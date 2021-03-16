[[vk::binding(0)]] Texture2D<float2> input_tex;
[[vk::binding(1)]] Texture2D<float2> orig_input_tex;
[[vk::binding(2)]] RWTexture2D<float2> output_tex;
[[vk::binding(3)]] cbuffer _ {
    float4 input_tex_size;
    int px_skip;
};


static const float gaussian_weights[5] = {
    1.0 / 16.0,
    1.0 / 4.0,
    3.0 / 8.0,
    1.0 / 4.0,
    1.0 / 16.0
};

[numthreads(8, 8, 1)]
void main(int2 px: SV_DispatchThreadID) {
    int2 tex_dim = int2(input_tex_size.xy);
    float center = orig_input_tex[px].x;
    //float center = texelFetch(input_tex, px, 0).r;

	float3 result = 0.0.xxx;
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            int2 offset = int2(x - 2, y - 2) * px_skip;
            int2 loc = px + offset;
            if (loc.x >= 0 && loc.y >= 0 && loc.x < tex_dim.x && loc.y < tex_dim.y) {
                float2 val = input_tex[loc].xy;
                float orig_val = orig_input_tex[loc].x;
                float w = 1;
                //w *= exp2(-(val - center) * (val - center) * 0.08 * (1.0 + log2(float(px_skip))));
                //w *= exp2(-(orig_val - center) * (orig_val - center) * 0.1 * (1.0 + log2(float(px_skip))));
                //w *= exp2(-(val - center) * (val - center) * 0.5);
                float diff = orig_val - center;
                w *= exp2(-diff * diff * 0.8);
                //w *= gaussian_weights[x] * gaussian_weights[y];
                w *= exp2(-dot(offset, offset) / (80 * 80));
                result += float3(val, 1) * w;
            }
        }
    }

    result.xy /= result.z;
	output_tex[px] = result.xy;
}
