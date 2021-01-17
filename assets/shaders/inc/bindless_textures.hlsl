#ifndef BINDLESS_TEXTURES_HLSL
#define BINDLESS_TEXTURES_HLSL

[[vk::binding(2, 1)]] Texture2D bindless_textures[];

#endif
