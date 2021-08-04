#ifndef PACK_UNPACK_HLSL
#define PACK_UNPACK_HLSL

float unpack_unorm(uint pckd, uint bitCount) {
	uint maxVal = (1u << bitCount) - 1;
	return float(pckd & maxVal) / maxVal;
}

uint pack_unorm(float val, uint bitCount) {
	uint maxVal = (1u << bitCount) - 1;
	return uint(clamp(val, 0.0, 1.0) * maxVal);
}

float pack_normal_11_10_11(float3 n) {
	uint pckd = 0;
	pckd += pack_unorm(n.x * 0.5 + 0.5, 11);
	pckd += pack_unorm(n.y * 0.5 + 0.5, 10) << 11;
	pckd += pack_unorm(n.z * 0.5 + 0.5, 11) << 21;
	return asfloat(pckd);
}

float3 unpack_normal_11_10_11(float pckd) {
	uint p = asuint(pckd);
	return normalize(float3(
		unpack_unorm(p, 11),
		unpack_unorm(p >> 11, 10),
		unpack_unorm(p >> 21, 11)
	) * 2.0 - 1.0);
}

float3 unpack_normal_11_10_11_no_normalize(float pckd) {
	uint p = asuint(pckd);
	return float3(
		unpack_unorm(p, 11),
		unpack_unorm(p >> 11, 10),
		unpack_unorm(p >> 21, 11)
	) * 2.0 - 1.0;
}

float3 unpack_normal_11_10_11_uint_no_normalize(uint p) {
	return float3(
		unpack_unorm(p, 11),
		unpack_unorm(p >> 11, 10),
		unpack_unorm(p >> 21, 11)
	) * 2.0 - 1.0;
}

uint pack_color_888(float3 color) {
    color = sqrt(color);
	uint pckd = 0;
	pckd += pack_unorm(color.x, 8);
	pckd += pack_unorm(color.y, 8) << 8;
	pckd += pack_unorm(color.z, 8) << 16;
    return pckd;
}

float3 unpack_color_888(uint p) {
	float3 color = float3(
		unpack_unorm(p, 8),
		unpack_unorm(p >> 8, 8),
		unpack_unorm(p >> 16, 8)
	);
    return color * color;
}

float2 octa_wrap( float2 v ) {
    return (1.0 - abs(v.yx)) * (step(0.0.xx, v.xy) * 2.0 - 1.0);
}
 
float2 octa_encode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        n.xy = octa_wrap(n.xy);
    }
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}

float3 octa_decode(float2 f)
{
    f = f * 2.0 - 1.0;
 
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3( f.x, f.y, 1.0 - abs( f.x ) - abs( f.y ) );
    float t = clamp(-n.z, 0.0, 1.0);
    //n.xy += n.xy >= 0.0 ? -t : t;
    n.xy -= (step(0.0, n.xy) * 2 - 1) * t;
    return normalize( n );
}

uint pack_2x16f_uint(float2 f) {
    return f32tof16(f.x) | (f32tof16(f.y) << 16u);
}

float2 unpack_2x16f_uint(uint u) {
    return float2(
    	f16tof32(u & 0xffff),
    	f16tof32((u >> 16) & 0xffff)
    );
}

#define RGB9E5_EXPONENT_BITS          5
#define RGB9E5_MANTISSA_BITS          9
#define RGB9E5_EXP_BIAS               15
#define RGB9E5_MAX_VALID_BIASED_EXP   31

#define MAX_RGB9E5_EXP               (RGB9E5_MAX_VALID_BIASED_EXP - RGB9E5_EXP_BIAS)
#define RGB9E5_MANTISSA_VALUES       (1<<RGB9E5_MANTISSA_BITS)
#define MAX_RGB9E5_MANTISSA          (RGB9E5_MANTISSA_VALUES-1)
#define MAX_RGB9E5                   ((float(MAX_RGB9E5_MANTISSA))/RGB9E5_MANTISSA_VALUES * (1<<MAX_RGB9E5_EXP))
#define EPSILON_RGB9E5               ((1.0/RGB9E5_MANTISSA_VALUES) / (1<<RGB9E5_EXP_BIAS))

float clamp_range_for_rgb9e5(float x) {
    return clamp(x, 0.0, MAX_RGB9E5);
}

int floor_log2(float x) {
    uint f = asuint(x);
    uint biasedexponent = (f & 0x7F800000u) >> 23;
    return int(biasedexponent) - 127;
}


// https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
uint float3_to_rgb9e5(float3 rgb) {
    float rc = clamp_range_for_rgb9e5(rgb.x);
    float gc = clamp_range_for_rgb9e5(rgb.y);
    float bc = clamp_range_for_rgb9e5(rgb.z);

    float maxrgb = max(rc, max(gc, bc));
    int exp_shared = max(-RGB9E5_EXP_BIAS-1, floor_log2(maxrgb)) + 1 + RGB9E5_EXP_BIAS;
    float denom = exp2(float(exp_shared - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS));

    int maxm = int(floor(maxrgb / denom + 0.5));
    if (maxm == MAX_RGB9E5_MANTISSA + 1) {
        denom *= 2;
        exp_shared += 1;
    }

    int rm = int(floor(rc / denom + 0.5));
    int gm = int(floor(gc / denom + 0.5));
    int bm = int(floor(bc / denom + 0.5));

    return (uint(rm) << (32 - 9))
        | (uint(gm) << (32 - 9 * 2))
        | (uint(bm) << (32 - 9 * 3))
        | uint(exp_shared);
}

uint bitfield_extract(uint value, uint offset, uint bits) {
    uint mask = (1u << bits) - 1u;
    return (value >> offset) & mask;
}

float3 rgb9e5_to_float3(uint v) {
    int exponent =
        int(bitfield_extract(v, 0, RGB9E5_EXPONENT_BITS)) - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS;
    float scale = exp2(float(exponent));

    return float3(
        float(bitfield_extract(v, 32 - RGB9E5_MANTISSA_BITS, RGB9E5_MANTISSA_BITS)) * scale,
        float(bitfield_extract(v, 32 - RGB9E5_MANTISSA_BITS * 2, RGB9E5_MANTISSA_BITS)) * scale,
        float(bitfield_extract(v, 32 - RGB9E5_MANTISSA_BITS * 3, RGB9E5_MANTISSA_BITS)) * scale
    );
}

// Used with B10G11R11_UFLOAT_PACK32 output
// The GPU will convert floating point values to 11_11_10 by _trimming_ the value,
// which often results in value loss and discoloration.
// By using this just before storing the value to an image, the conversion becomes rounding.
float3 prequant_shift_11_11_10(float3 v) {
    static const float3 F_11_11_10_MANTISSA_BITS = float3(6, 6, 5);

    const float3 exponent = ceil(log2(v));

    // Add a 0.5 just below what the format can represent
    return v + exp2(exponent - F_11_11_10_MANTISSA_BITS - 2);
}

#endif