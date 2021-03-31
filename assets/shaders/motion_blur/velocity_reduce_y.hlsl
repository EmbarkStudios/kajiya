[[vk::binding(0)]] Texture2D<float2> input_tex;
[[vk::binding(1)]] RWTexture2D<float2> output_tex;

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
	float3 largest_velocity = float3(0, 0, 0);

	for (int y = 0; y < 16; ++y) {
		float2 v = input_tex[px * int2(1, 16) + int2(0, y)];
		float m2 = dot(v, v);
		largest_velocity = m2 > largest_velocity.z ? float3(v, m2) : largest_velocity;
	}

    output_tex[px] = largest_velocity.xy;
}
