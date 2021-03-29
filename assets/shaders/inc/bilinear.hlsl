// From "Fast Denoising with Self Stabilizing Recurrent Blurs"

struct Bilinear {
    float2 origin;
    float2 weights;

    int2 px0() {
        return int2(origin);
    }

    int2 px1() {
        return int2(origin) + int2(1, 0);
    }

    int2 px2() {
        return int2(origin) + int2(0, 1);
    }

    int2 px3() {
        return int2(origin) + int2(1, 1);
    }
};

Bilinear get_bilinear_filter(float2 uv, float2 tex_size) {
	Bilinear result;
	result.origin = trunc(uv * tex_size - 0.5);
	result.weights = frac(uv * tex_size - 0.5);
	return result;
}

float4 get_bilinear_custom_weights(Bilinear f, float4 custom_weights) {
	float4 weights;
	weights.x = (1.0 - f.weights.x) * (1.0 - f.weights.y);
	weights.y = f.weights.x * (1.0 - f.weights.y);
	weights.z = (1.0 - f.weights.x) * f.weights.y;
	weights.w = f.weights.x * f.weights.y;
	return weights * custom_weights;
}

float4 apply_bilinear_custom_weights(float4 s00, float4 s10, float4 s01, float4 s11, float4 w, bool normalize = true) {
	float4 r = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
	return r * (normalize ? rcp(dot(w, 1.0)) : 1.0);
}
