#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(binding = 0, set = 0) uniform sampler2D font_texture;

void main() { outColor = inColor.rgba * texture(font_texture, inUV); }

