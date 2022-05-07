#ifndef MORTON_HLSL
#define MORTON_HLSL

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
uint compact_1_by_1(uint x) {
    x &= 0x55555555u;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1u)) & 0x33333333u; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2u)) & 0x0f0f0f0fu; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4u)) & 0x00ff00ffu; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8u)) & 0x0000ffffu; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

uint2 decode_morton_2d(uint c) {
    return uint2(compact_1_by_1(c), compact_1_by_1(c >> 1u));
}

#endif  // MORTON_HLSL
