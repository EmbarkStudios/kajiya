#include "rendertoy::shaders/view_constants.inc"
#include "rendertoy::shaders/random.inc"
#include "rendertoy::shaders/sampling.inc"
#include "rtoy-samples::shaders/inc/uv.inc"
#include "rtoy-samples::shaders/inc/pack_unpack.inc"
#include "rtoy-samples::shaders/inc/brdf.inc"
#include "sdf_consts.inc"

uniform texture2D gbuffer;
uniform texture2D aoTex;
//uniform texture2D shadowsTex;

uniform texture2D skyLambertTex;
uniform texture2D skyOctaTex;
uniform texture2D skyTex;

uniform sampler linear_clamp_sampler;

uniform restrict writeonly image2D outputTex;

layout(std140) uniform globals {
    vec4 gbuffer_size;
    vec4 skyOctaTex_size;
    vec4 outputTex_size;
};

layout(std430) buffer constants {
    ViewConstants view_constants;
    vec4 light_dir_pad;
    uint frame_idx;
};

vec3 sample_quantized_environment_light(vec3 dir) {
    dir = normalize(dir);
    return texelFetch(skyOctaTex, ivec2(skyOctaTex_size.xy * octa_encode(dir)), 0).rgb;
}

vec3 sample_lambert_convolved_environment_light(vec3 dir) {
    dir = normalize(dir);
    return texelFetch(skyLambertTex, ivec2(skyOctaTex_size.xy * octa_encode(dir)), 0).rgb;
}

layout (local_size_x = 8, local_size_y = 8) in;
void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = get_uv(outputTex_size);
    vec4 ray_origin_cs = vec4(uv_to_cs(uv), 1.0, 1.0);
    vec4 ray_origin_ws = view_constants.view_to_world * (view_constants.sample_to_view * ray_origin_cs);
    ray_origin_ws /= ray_origin_ws.w;

    vec4 ray_dir_cs = vec4(uv_to_cs(uv), 0.0, 1.0);
    vec4 ray_dir_ws = view_constants.view_to_world * (view_constants.sample_to_view * ray_dir_cs);
    vec3 v = -normalize(ray_dir_ws.xyz);

    vec4 gbuffer = texelFetch(gbuffer, pix, 0);

    vec3 result = vec3(0, 0, 0);
    vec3 sun_color = sample_quantized_environment_light(light_dir_pad.xyz) * vec3(1.3, 1, 0.7);

    if (gbuffer.a == 0.0) {
        result = textureLod(sampler2D(skyTex, linear_clamp_sampler), uv, 0.0).rgb;
    } else {
        vec3 normal = unpack_normal_11_10_11(gbuffer.x);
        vec3 albedo = unpack_color_888(floatBitsToUint(gbuffer.z));
        vec3 env_color = sample_lambert_convolved_environment_light(normal);

        vec4 ssgi = texelFetch(aoTex, pix, 0);
        //ssgi = vec4(0, 0, 0, 1);

        result += albedo * env_color * ssgi.a;
        result += albedo * ssgi.rgb;

        float shadows = 1;//texelFetch(shadowsTex, pix, 0).r;
        float ndotl = max(0, dot(normal, light_dir_pad.xyz));
        result += albedo * ndotl * shadows * sun_color;

        //result = albedo;
    }

    const bool visualize_volume = true;
    if (visualize_volume) {
        vec3 entry = ray_origin_ws.xyz; {
            vec3 d = (HSIZE.xxx - entry * sign(v)) / -max(vec3(1e-5), abs(v));
            entry += max(0.0, max(max(d.x, d.y), d.z)) * -v;
        }

        vec3 exit = entry;
        if (gbuffer.a != 0.0) {
            vec4 hit_cs = vec4(uv_to_cs(uv), gbuffer.w, 1.0);
            vec4 hit_vs = view_constants.sample_to_view * hit_cs;
            vec4 hit_ws = view_constants.view_to_world * hit_vs;
            hit_ws /= hit_ws.w;
            exit = hit_ws.xyz;
        } else {
            vec3 d = (-HSIZE.xxx - exit * sign(v)) / max(vec3(1e-5), abs(v));
            exit += max(0.0, -max(max(d.x, d.y), d.z)) * -v;
        }

        result.r += (1.0 - exp(-0.04 * length(entry - exit)));
    }

    {
        uint seed0 = hash(hash(pix.x) ^ pix.y);
        float rnd = rand_float(seed0);
        result.rgb += (rnd - 0.5) / 512.0;
    }

    imageStore(outputTex, pix, vec4(result, 1));
}
