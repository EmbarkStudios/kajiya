#version 430
#include "hlsl_to_glsl.glsl"
#include "math.hlsl"

uniform sampler2D input_texture;
uniform float input_ev;
in float2 input_uv;
out float4 output_rgba;

#define DECLARE_BEZOLD_BRUCKE_LUT uniform sampler1D bezold_brucke_lut
#define SAMPLE_BEZOLD_BRUCKE_LUT(coord) textureLod(bezold_brucke_lut, (coord), 0).xy

struct ShaderInput {
    float3 stimulus;
    float2 uv;
};

ShaderInput prepare_shader_input() {
    ShaderInput shader_input;
    shader_input.stimulus = exp2(input_ev) * max(0.0.xxx, textureLod(input_texture, input_uv, 0).rgb);
    shader_input.uv = input_uv;
    return shader_input;
}

#define SHADER_MAIN_FN output_rgba = float4(compress_stimulus(prepare_shader_input()), 1.0);
