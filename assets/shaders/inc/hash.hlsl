#ifndef HASH_HLSL
#define HASH_HLSL

#include "math_const.hlsl"

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

uint hash4(uint4 v) {
	return hash_combine2(v.x, hash3(v.yzw));
}

float uint_to_u01_float(uint h) {
	static const uint mantissaMask = 0x007FFFFFu;
	static const uint one = 0x3F800000u;

	h &= mantissaMask;
	h |= one;

	float  r2 = asfloat( h );
	return r2 - 1.0;
}

#endif 