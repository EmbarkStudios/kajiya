#ifndef IRCACHE_SAMPLER_COMMON_INC_HLSL
#define IRCACHE_SAMPLER_COMMON_INC_HLSL

static const uint SAMPLER_SEQUENCE_LENGTH = 1024;

struct SampleParams {
    uint value;

    static SampleParams from_spf_entry_sample_frame(uint samples_per_frame, uint entry_idx, uint sample_idx, uint frame_idx) {
        const uint PERIOD = IRCACHE_OCTA_DIMS2 / samples_per_frame;

        uint xy = sample_idx * PERIOD + (frame_idx % PERIOD);

        // Checkerboard
        xy ^= (xy & 4u) >> 2u;

        SampleParams res;
        res.value = xy + ((frame_idx << 16u) ^ (entry_idx)) * IRCACHE_OCTA_DIMS2;

        return res;
    }

    static SampleParams from_raw(uint raw) {
        SampleParams res;
        res.value = raw;
        return res;
    }

    uint raw() {
        return value;
    }

    uint octa_idx() {
        return value % IRCACHE_OCTA_DIMS2;
    }

    uint2 octa_quant() {
        uint oi = octa_idx();
        return uint2(oi % IRCACHE_OCTA_DIMS, oi / IRCACHE_OCTA_DIMS);
    }

    uint rng() {
        return hash1(value >> 4u);
    }

    float2 octa_uv() {
        const uint2 oq = octa_quant();
        const uint r = rng();
        const float2 urand = r2_sequence(r % SAMPLER_SEQUENCE_LENGTH);
        return (float2(oq) + urand) / 4.0;
    }

    // TODO: tackle distortion
    float3 direction() {
        return octa_decode(octa_uv());
    }
};

#endif  // IRCACHE_SAMPLER_COMMON_INC_HLSL
