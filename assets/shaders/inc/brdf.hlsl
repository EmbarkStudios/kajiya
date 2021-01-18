#include "math.hlsl"

static const uint2 BRDF_FG_LUT_DIMS = uint2(64, 64);
static const float2 BRDF_FG_LUT_UV_SCALE = (BRDF_FG_LUT_DIMS - 1.0) / BRDF_FG_LUT_DIMS;
static const float2 BRDF_FG_LUT_UV_BIAS = 0.5.xx / BRDF_FG_LUT_DIMS;

struct BrdfValue {
    float3 value_over_pdf;
    float pdf;

    float3 transmission_fraction;

    float3 value() {
        return value_over_pdf * pdf;
    }

    static BrdfValue invalid() {
        BrdfValue res;
        res.value_over_pdf = 0.0;
        res.pdf = 0.0;
        res.transmission_fraction = 0.0;
        return res;
    }
};

struct BrdfSample: BrdfValue {
    float3 wi;

    static BrdfSample invalid() {
        BrdfSample res;
        res.value_over_pdf = 0.0;
        res.pdf = 0.0;
        res.wi = float3(0.0, 0.0, -1.0);
        res.transmission_fraction = 0.0;
        return res;
    }

    bool is_valid() {
        return wi.z > 1e-6;
    }
};

struct DiffuseBrdf {
    float3 albedo;
    //float3 emission;

    BrdfSample sample(float3 _wo, float2 urand) {
        float phi = urand.x * M_TAU;
        float cos_theta = sqrt(max(0.0, 1.0 - urand.y));
        float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

        BrdfSample res;
        float sin_phi = sin(phi);
        float cos_phi = cos(phi);

        res.wi = float3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
        res.pdf = M_FRAC_1_PI;
        res.value_over_pdf = albedo;
        res.transmission_fraction = 0.0;

        return res;
	}

    BrdfValue evaluate(float3 _wo, float3 wi) {
		BrdfValue res;
		res.pdf = wi.z > 0.0 ? M_FRAC_1_PI : 0.0;
		res.value_over_pdf = wi.z > 0.0 ? albedo : 0.0.xxx;
        res.transmission_fraction = 0.0;
		return res;
	}
};

float3 eval_fresnel_schlick(float3 f0, float3 f90, float cos_theta) {
    return lerp(f0, f90, pow(max(0.0, 1.0 - cos_theta), 5));
}

struct SpecularBrdf {
    float roughness;
    float3 albedo;
    //float3 emission;

	static float ggx_ndf(float a2, float ndotm) {
		float denom_sqrt = ndotm * ndotm * (a2 - 1.0) + 1.0;
		return a2 / (M_PI * denom_sqrt * denom_sqrt);
	}

    float g_smith_ggx_correlated(float ndotv, float ndotl, float ag) {
    	float ag2 = ag * ag;

    	float lambda_v = ndotl * sqrt((-ndotv * ag2 + ndotv) * ndotv + ag2);
    	float lambda_l = ndotv * sqrt((-ndotl * ag2 + ndotl) * ndotl + ag2);

    	return 2.0 * ndotl * ndotv / (lambda_v + lambda_l);
    }

    BrdfSample sample(float3 wo, float2 urand) {
        const float a2 = roughness * roughness;

		const float cos2_theta = (1 - urand.x) / (1 - urand.x + a2 * urand.x);
		const float cos_theta = sqrt(cos2_theta);
		const float phi = M_TAU * urand.y;

		const float sin_theta = sqrt(max(0.0, 1.0 - cos2_theta));
		const float3 m = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
		const float3 wi = -wo + m * dot(wo, m) * 2.0;

		if (m.z <= 0.0 || wi.z <= 0.0 || wo.z <= 0.0) {
			return BrdfSample::invalid();
		}

		const float pdf_h = ggx_ndf(a2, cos_theta) * cos_theta;

		// Change of variables from the half-direction space to regular lighting geometry.
		const float jacobian = 1.0 / (4.0 * dot(wi, m));

        const float3 fresnel = eval_fresnel_schlick(albedo, 1.0, dot(m, wi));

        BrdfSample res;
		res.pdf = pdf_h * jacobian / wi.z;
		res.value_over_pdf =
            fresnel
            / (cos_theta * jacobian)
			* g_smith_ggx_correlated(wo.z, wi.z, roughness)
            / (4 * wo.z);
		res.wi = wi;
        res.transmission_fraction = 1.0.xxx - fresnel;

		return res;
	}

    BrdfValue evaluate(float3 wo, float3 wi) {
        if (wi.z <= 0.0 || wo.z <= 0.0) {
            return BrdfValue::invalid();
        }

        const float a2 = roughness * roughness;

        const float3 m = normalize(wo + wi);

        const float cos_theta = m.z;
        const float pdf_h_denom_sqrt = 1.0 + (-1.0 + a2) * cos_theta * cos_theta;
        const float pdf_h = ggx_ndf(a2, cos_theta) * cos_theta;
        const float jacobian = 1.0 / (4.0 * dot(wi, m));

        const float3 fresnel = eval_fresnel_schlick(albedo, 1.0, dot(m, wi));

        BrdfValue res;
        res.pdf = pdf_h * jacobian / wi.z;
        res.value_over_pdf =
           fresnel
            / (cos_theta * jacobian)
            *	g_smith_ggx_correlated(wo.z, wi.z, roughness)
            / (4 * wo.z);
        res.transmission_fraction = 1.0.xxx - fresnel;

        return res;
	}
};

float roughness_to_perceptual_roughness(float r) {
    return sqrt(r);
}

float perceptual_roughness_to_roughness(float r) {
    return r * r;
}

float3 calculate_metalness_albedo_boost(float metalness, float3 diffuse_albedo) {
    static const float a0 = 0.9203;
    static const float a1 = 3.21;
    static const float b1 =-8.412;
    static const float e1 = 0.8295;
    static const float e3 = 1.81;

    const float lobe_offset = (metalness - 0.5) * (metalness - 0.5);
    const float envelope = 0.25 - lobe_offset;
    const float lobe = envelope * exp(a1 * lobe_offset);

    const float3 y = diffuse_albedo;
    const float3 y3 = y * y * y;
    const float3 ramp = e1 * y + e3 * y3;

    return 1 + lobe * ramp;
}
