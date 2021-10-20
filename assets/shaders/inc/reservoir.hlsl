#ifndef RESERVOIR_HLSL
#define RESERVOIR_HLSL

#include "hash.hlsl"

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
        const float dart = uint_to_u01_float(hash1_mut(rng));
        const float prob = w / this.w_sum;

        if (prob >= dart) {
            this.payload = sample_payload;
            return true;
        } else {
            return false;
        }
    }
};

#endif // RESERVOIR_HLSL
