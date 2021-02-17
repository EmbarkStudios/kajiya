#ifndef GBUFFER_HLSL
#define GBUFFER_HLSL

struct GbufferData;

struct GbufferDataPacked {
    uint4 data0;

    static GbufferDataPacked from_uint4(uint4 data0) {
        GbufferDataPacked res;
        res.data0 = data0;
        return res;
    }

    GbufferData unpack();
    float3 unpack_normal();
    float3 unpack_albedo();
};

struct GbufferData {
    float3 albedo;
    float3 normal;
    float roughness;
    float metalness;

    GbufferDataPacked pack();
};


GbufferDataPacked GbufferData::pack() {
    float4 res = 0.0.xxxx;
    res.x = asfloat(pack_color_888(albedo));
    res.y = pack_normal_11_10_11(normal);
    res.z = roughness * roughness;      // UE4 remap
    res.w = metalness;

   GbufferDataPacked packed;
   packed.data0 = asuint(res);
   return packed;
}

GbufferData GbufferDataPacked::unpack() {
    GbufferData res;
    res.albedo = unpack_albedo();
    res.normal = unpack_normal();
    res.roughness = sqrt(asfloat(data0.z));
    res.metalness = asfloat(data0.w);
    return res;
}

float3 GbufferDataPacked::unpack_normal() {
    return unpack_normal_11_10_11(asfloat(data0.y));
}

float3 GbufferDataPacked::unpack_albedo() {
    return unpack_color_888(data0.x);
}

#endif