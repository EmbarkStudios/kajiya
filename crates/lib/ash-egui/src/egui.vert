#version 450

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;

layout(push_constant) uniform PushConstants { vec2 screen_size; }
pushConstants;

float sRGB_OETF(float a) {
	return .04045f < a ? pow((a + .055f) / 1.055f, 2.4f) : a / 12.92f;
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(
        sRGB_OETF(srgba.r),
        sRGB_OETF(srgba.g),
        sRGB_OETF(srgba.b),
        srgba.a);
}

void main() {
  gl_Position =
      vec4(2.0 * inPos.x / pushConstants.screen_size.x - 1.0,
           2.0 * inPos.y / pushConstants.screen_size.y - 1.0, 0.0, 1.0);
  outColor = linear_from_srgba(inColor);
  outUV = inUV;
}
