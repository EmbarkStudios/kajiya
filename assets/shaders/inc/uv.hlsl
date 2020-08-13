float2 get_uv(int2 pix, float4 texSize) {
	return (float2(pix) + 0.5) * texSize.zw;
}

float2 get_uv(float2 pix, float4 texSize) {
	return (pix + 0.5) * texSize.zw;
}

float2 cs_to_uv(float2 cs)
{
	return cs * float2(0.5, -0.5) + float2(0.5, 0.5);
}

float2 uv_to_cs(float2 uv)
{
	return (uv - 0.5.xx) * float2(2, -2);
}
