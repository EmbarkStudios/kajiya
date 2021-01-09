[[vk::binding(2, 1)]] Texture2D material_textures[];

struct PsIn {
    [[vk::location(0)]] float4 color: COLOR0;
};

float4 main(PsIn ps/*, float4 cs_pos: SV_Position*/): SV_TARGET {
    return ps.color;
}
