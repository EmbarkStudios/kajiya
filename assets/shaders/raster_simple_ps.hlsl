SamplerState sampler_lnc;
[[vk::binding(0, 3)]] Texture2D material_textures[];

struct PsIn {
    [[vk::location(0)]] float4 color: COLOR0;
};

float4 main(PsIn ps/*, float4 cs_pos: SV_Position*/): SV_TARGET {
    Texture2D tex = material_textures[7];
    float4 col = tex.Sample(sampler_lnc, ps.color.xy);
    return col;

    return ps.color;
}
