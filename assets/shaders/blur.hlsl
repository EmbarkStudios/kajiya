Texture2D<float4> input_tex;
RWTexture2D<float4> output_tex;

static const uint kernel_radius = 5;
static const uint group_width = 64;
static const uint vblur_window_size = (group_width + kernel_radius) * 2;

#define Color float4
#define Color_swizzle xyzw

float gaussian_wt(float dst_px, float src_px) {
    float px_off = (dst_px + 0.5) * 2 - (src_px + 0.5);
    float sigma = kernel_radius * 0.5;
    return exp(-px_off * px_off / (sigma * sigma));
}

Color vblur(int2 dst_px, int2 src_px) {
	Color res = 0;
    float wt_sum = 0;

	for (uint y = 0; y <= kernel_radius * 2; ++y) {
        float wt = gaussian_wt(dst_px.y, src_px.y + y);
		res += input_tex[src_px + int2(0, y)].Color_swizzle * wt;
        wt_sum += wt;
	}

	return res / wt_sum;
}

groupshared Color vblur_out[vblur_window_size];

void vblur_into_shmem(int2 dst_px, uint xfetch, uint2 group_id) {
    int2 src_px = group_id * int2(group_width * 2, 2) + int2(xfetch - kernel_radius, -kernel_radius);
    vblur_out[xfetch] = vblur(dst_px, src_px);
}

[numthreads(group_width, 1, 1)]
void main(uint2 px: SV_DispatchThreadID, uint2 px_within_group: SV_GroupThreadID, uint2 group_id: SV_GroupID) {
	for (int xfetch = px_within_group.x; xfetch < vblur_window_size; xfetch += group_width) {
        vblur_into_shmem(px, xfetch, group_id);
	}

    GroupMemoryBarrierWithGroupSync();

    float4 res = 0;
    float wt_sum = 0;

    for (uint x = 0; x <= kernel_radius * 2; ++x) {
        float wt = gaussian_wt(px.x, px.x * 2 + x - kernel_radius);
        res.Color_swizzle += vblur_out[px_within_group.x * 2 + x] * wt;
        wt_sum += wt;
    }
    res /= wt_sum;

    output_tex[px] = res;
}
