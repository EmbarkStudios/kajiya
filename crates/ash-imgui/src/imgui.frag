#version 430 core

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_col;

layout(set = 0, binding = 0) uniform sampler2D g_tex;

layout(location = 0) out vec4 o_col;

void main() {
    o_col = v_col*texture(g_tex, v_uv);
}
