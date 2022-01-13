#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

layout(push_constant) uniform PushConstants { vec2 screen_size; }
pushConstants;

void main() {
  gl_Position =
      vec4(2.0 * inPos.x / pushConstants.screen_size.x - 1.0,
           2.0 * inPos.y / pushConstants.screen_size.y - 1.0, 0.0, 1.0);
  outColor = inColor;
  outUV = inUV;
}
