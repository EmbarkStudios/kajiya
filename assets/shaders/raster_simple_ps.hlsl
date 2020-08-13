struct PsIn {
    [[vk::location(0)]] float4 color: COLOR0;
};


float4 main(PsIn ps): SV_TARGET {
    return ps.color;
}
