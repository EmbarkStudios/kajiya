#include "inc/frame_constants.hlsl"
#include "inc/pack_unpack.hlsl"
#include "inc/uv.hlsl"
#include "inc/tonemap.hlsl"

Texture2D<float4> gbuffer_tex;
Texture2D<float> depth_tex;
RWTexture2D<float4> output_tex;

#define PI 3.14159
#define TWO_PI 6.28318

static const float3 ambient_light = 0.1;

float g_smith_ggx_correlated(float ndotv, float ndotl, float ag) {
	float ag2 = ag * ag;

	float lambda_v = ndotl * sqrt((-ndotv * ag2 + ndotv) * ndotv + ag2);
	float lambda_l = ndotv * sqrt((-ndotl * ag2 + ndotl) * ndotl + ag2);

	return 2.0 * ndotl * ndotv / (lambda_v + lambda_l);
}

float d_ggx(float ndotm, float a) {
    float a2 = a * a;
	float denom_sqrt = ndotm * ndotm * (a2 - 1.0) + 1.0;
	return a2 / (PI * denom_sqrt * denom_sqrt);
}

[numthreads(8, 8, 1)]
void main(in uint2 pix : SV_DispatchThreadID) {
    float4 gbuffer = gbuffer_tex[pix];
    if (all(gbuffer == 0.0.xxxx)) {
        output_tex[pix] = float4(ambient_light, 1.0);
        return;
    }

    float z_over_w = depth_tex[pix];

    float4 output_tex_size = float4(1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0);
    float2 uv = get_uv(pix, output_tex_size);

    ViewConstants view_constants = frame_constants.view_constants;
    float4 ray_dir_cs = float4(uv_to_cs(uv), 0.0, 1.0);
    float4 ray_dir_vs = mul(view_constants.sample_to_view, ray_dir_cs);
    float4 ray_dir_ws = mul(view_constants.view_to_world, ray_dir_vs);

    float3 v = -normalize(ray_dir_ws.xyz);
    float3 v_vs = -normalize(ray_dir_vs.xyz);

    float4 pt_cs = float4(uv_to_cs(uv), z_over_w, 1.0);
    float4 pt_ws = mul(view_constants.view_to_world, mul(view_constants.sample_to_view, pt_cs));
    pt_ws /= pt_ws.w;

    float3 albedo = unpack_color_888(asuint(gbuffer.x));
    float3 normal = unpack_normal_11_10_11(gbuffer.y);
    float roughness = sqrt(gbuffer.z);
    float metalness = gbuffer.w;

    float4 res = 0.0.xxxx;

    float3 l = normalize(float3(1, 1, 1));
    float3 h = normalize(l + v);

    float ndoth = abs(dot(normal, h));
    float ldoth = abs(dot(l, h));
    float ndotv = max(0.0, dot(normal, v));
    float ndotl = max(0.0, dot(normal, l));

    float3 f0 = lerp(0.04, albedo, metalness);
    float schlick = pow(max(0.0, 1.0 - ldoth), 5.0);
    float3 fr = lerp(f0, 1.0.xxx, schlick);

    float brdf_d = d_ggx(ndoth, roughness);
    float brdf_g = g_smith_ggx_correlated(ndotv, ndotl, roughness);

    float3 diffuse_color = max(0.0, 1.0 - metalness) * albedo;
    float3 diffuse = diffuse_color * ndotl;
    float3 spec = brdf_d * brdf_g / PI;

    float3 radiance = lerp(diffuse, spec, fr);
    float3 ambient = ambient_light * albedo;

    float3 light_radiance = 3.0;
    //res.xyz += normal * 0.5 + 0.5;
    res.xyz += radiance * light_radiance + ambient;
    //res.xyz += albedo;
    //res.xyz += metalness;
    //res.xyz = metalness;
    //res.xyz = 0.0001 / z_over_w;
    //res.xyz = frac(pt_ws.xyz * 10.0);
    //res.xyz = brdf_d * 0.1;

    //res.xyz = 1.0 - exp(-res.xyz);
    res.xyz = neutral_tonemap(res.xyz);

    output_tex[pix] = res;
}
