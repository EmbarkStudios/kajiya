#ifndef HASH_HLSL
#define HASH_HLSL

uint hash1(uint x) {
	x += (x << 10u);
	x ^= (x >>  6u);
	x += (x <<  3u);
	x ^= (x >> 11u);
	x += (x << 15u);
	return x;
}

uint hash1_mut(inout uint h) {
    uint res = h;
    h = hash1(h);
    return res;
}

uint hash_combine2(uint x, uint y) {
    static const uint M = 1664525u, C = 1013904223u;
    uint seed = (x * M + y + C) * M;

    // Tempering (from Matsumoto)
    seed ^= (seed >> 11u);
    seed ^= (seed << 7u) & 0x9d2c5680u;
    seed ^= (seed << 15u) & 0xefc60000u;
    seed ^= (seed >> 18u);
    return seed;
}

uint hash2(uint2 v) {
	return hash_combine2(v.x, hash1(v.y));
}

uint hash3(uint3 v) {
	return hash_combine2(v.x, hash2(v.yz));
}

float uint_to_u01_float(uint h) {
	static const uint mantissaMask = 0x007FFFFFu;
	static const uint one = 0x3F800000u;

	h &= mantissaMask;
	h |= one;

	float  r2 = asfloat( h );
	return r2 - 1.0;
}

float radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 hammersley(uint i, uint n) {
    return float2(float(i + 1) / n, radical_inverse_vdc(i + 1));
}

#endif 