#include "../inc/math.hlsl"

static const float SURFEL_TET_BASIS_SPOKE_H = 0.5;
static const float3 SURFEL_TET_BASIS[4] = {
    float3(0, 0, 1),
    float3(float2(cos(0 * M_TAU / 3), sin(0 * M_TAU / 3)) * sqrt(1 - pow(SURFEL_TET_BASIS_SPOKE_H, 2)), SURFEL_TET_BASIS_SPOKE_H),
    float3(float2(cos(1 * M_TAU / 3), sin(1 * M_TAU / 3)) * sqrt(1 - pow(SURFEL_TET_BASIS_SPOKE_H, 2)), SURFEL_TET_BASIS_SPOKE_H),
    float3(float2(cos(2 * M_TAU / 3), sin(2 * M_TAU / 3)) * sqrt(1 - pow(SURFEL_TET_BASIS_SPOKE_H, 2)), SURFEL_TET_BASIS_SPOKE_H)
};

float3 calc_surfel_tet_basis(float3 normal)[4] {
    float3 surfel_tet_basis[4];
    {
        const float3x3 surfel_xform_basis = build_orthonormal_basis(normal);
        [unroll]
        for (int i = 0; i < 4; ++i) {
            surfel_tet_basis[i] = mul(surfel_xform_basis, SURFEL_TET_BASIS[i]);
        }
    }

    return surfel_tet_basis;
}