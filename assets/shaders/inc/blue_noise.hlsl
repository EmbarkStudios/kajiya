#ifndef BLUE_NOISE_HLSL
#define BLUE_NOISE_HLSL

#include "quasi_random.hlsl"
#include "bindless_textures.hlsl"

// The source texture is RGBA8, and the output here is quantized to [0.5/256 .. 255.5/256]
float4 blue_noise_for_pixel(uint2 px, uint n) {
    const uint2 tex_dims = uint2(256, 256);
    const uint2 offset = r2_sequence(n) * tex_dims;

    return bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][
        (px + offset) % tex_dims
    ] * 255.0 / 256.0 + 0.5 / 256.0;
}

// ----
// https://crates.io/crates/blue-noise-sampler

#define DEFINE_BLUE_NOISE_SAMPLER_BINDINGS(b0, b1, b2) \
    [[vk::binding(b0)]] StructuredBuffer<uint> ranking_tile_buf; \
    [[vk::binding(b1)]] StructuredBuffer<uint> scambling_tile_buf; \
    [[vk::binding(b2)]] StructuredBuffer<uint> sobol_buf; \
    float blue_noise_sampler(int pixel_i,int pixel_j,int sampleIndex,int sampleDimension) { \
        return blue_noise_sampler(pixel_i, pixel_j, sampleIndex, sampleDimension, ranking_tile_buf, scambling_tile_buf, sobol_buf); \
    }

float blue_noise_sampler(
    int pixel_i,
    int pixel_j,
    int sampleIndex,
    int sampleDimension,
    StructuredBuffer<uint> ranking_tile_buf,
    StructuredBuffer<uint> scambling_tile_buf,
    StructuredBuffer<uint> sobol_buf
) {
	// wrap arguments
	pixel_i = pixel_i & 127;
	pixel_j = pixel_j & 127;
	sampleIndex = sampleIndex & 255;
	sampleDimension = sampleDimension & 255;

	// xor index based on optimized ranking
	// jb: 1spp blue noise has all 0 in ranking_tile_buf so we can skip the load
	int rankedSampleIndex = sampleIndex ^ ranking_tile_buf[sampleDimension + (pixel_i + pixel_j*128)*8];

	// fetch value in sequence
	int value = sobol_buf[sampleDimension + rankedSampleIndex*256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ scambling_tile_buf[(sampleDimension%8) + (pixel_i + pixel_j*128)*8];

	// convert to float and return
	float v = (0.5f+value)/256.0f;
	return v;
}

// ----


#endif