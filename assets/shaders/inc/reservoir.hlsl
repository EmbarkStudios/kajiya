#ifndef RESERVOIR_HLSL
#define RESERVOIR_HLSL

#include "hash.hlsl"

struct Reservoir1spp {
    float w_sum;
    float w_sel;
    float M;
    float W;

    static Reservoir1spp create() {
        Reservoir1spp res;
        res.w_sum = 0;
        res.w_sel = 0;
        res.M = 0;
        res.W = 0;
        return res;
    }

    static Reservoir1spp from_raw(float4 raw) {
        Reservoir1spp res;
        res.w_sum = raw.x;
        res.w_sel = raw.y;
        res.M = raw.z;
        res.W = raw.w;
        return res;
    }

    float4 as_raw() {
        return float4(w_sum, w_sel, M, W);
    }

    bool update(float w, inout uint rng) {
        this.w_sum += w;
        const float dart = uint_to_u01_float(hash1_mut(rng));
        const float prob = w / this.w_sum;

        if (prob >= dart) {
            this.w_sel = w;
            return true;
        } else {
            return false;
        }
    }
};

#endif // RESERVOIR_HLSL
