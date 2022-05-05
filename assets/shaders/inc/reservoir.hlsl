#ifndef RESERVOIR_HLSL
#define RESERVOIR_HLSL

#include "hash.hlsl"

struct Reservoir1sppStreamState {
    float p_q_sel;
    float M_sum;

    static Reservoir1sppStreamState create() {
        Reservoir1sppStreamState res;
        res.p_q_sel = 0;
        res.M_sum = 0;
        return res;
    }
};

struct Reservoir1spp {
    float w_sum;
    uint payload;
    float M;
    float W;

    static Reservoir1spp create() {
        Reservoir1spp res;
        res.w_sum = 0;
        res.payload = 0;
        res.M = 0;
        res.W = 0;
        return res;
    }

    static Reservoir1spp from_raw(float4 raw) {
        Reservoir1spp res;
        res.w_sum = raw.x;
        res.payload = asuint(raw.y);
        res.M = raw.z;
        res.W = raw.w;
        return res;
    }

    float4 as_raw() {
        return float4(w_sum, asfloat(payload), M, W);
    }

    bool update(float w, uint sample_payload, inout uint rng) {
        this.w_sum += w;
        this.M += 1;
        const float dart = uint_to_u01_float(hash1_mut(rng));
        const float prob = w / this.w_sum;

        if (prob >= dart) {
            this.payload = sample_payload;
            return true;
        } else {
            return false;
        }
    }

    bool update_with_stream(
        Reservoir1spp r,
        float p_q,
        float weight,
        inout Reservoir1sppStreamState stream_state,
        uint sample_payload,
        inout uint rng
    ) {
        stream_state.M_sum += r.M;

        if (update(p_q * weight * r.W * r.M, sample_payload, rng)) {
            stream_state.p_q_sel = p_q;
            return true;
        } else {
            return false;
        }
    }

    void init_with_stream(
        float p_q,
        float weight,
        inout Reservoir1sppStreamState stream_state,
        uint sample_payload
    ) {
        payload = sample_payload;
        w_sum = p_q * weight;
        M = weight != 0 ? 1 : 0;
        W = weight;

        stream_state.p_q_sel = p_q;
        stream_state.M_sum = M;
    }

    void finish_stream(Reservoir1sppStreamState state) {
        M = state.M_sum;
        W = (1.0 / max(1e-5, state.p_q_sel)) * (w_sum / max(1e-5, M));
    }
};

#endif // RESERVOIR_HLSL
