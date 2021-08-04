#ifndef MESH_HLSL
#define MESH_HLSL

struct VertexPacked {
	float4 data0;
};

struct Mesh {
    uint vertex_core_offset;
    uint vertex_uv_offset;
    uint vertex_mat_offset;
    uint vertex_aux_offset;
    uint vertex_tangent_offset;
    uint mat_data_offset;
    uint index_offset;
};

struct Vertex {
    float3 position;
    float3 normal;
};

float3 unpack_unit_direction_11_10_11(uint pck) {
    return float3(
        float(pck & ((1u << 11u)-1u)) * (2.0f / float((1u << 11u)-1u)) - 1.0f,
        float((pck >> 11u) & ((1u << 10u)-1u)) * (2.0f / float((1u << 10u)-1u)) - 1.0f,
        float((pck >> 21u)) * (2.0f / float((1u << 11u)-1u)) - 1.0f
    );
}

Vertex unpack_vertex(VertexPacked p) {
    Vertex res;
    res.position = p.data0.xyz;
    res.normal = unpack_unit_direction_11_10_11(asuint(p.data0.w));
    return res;
}

static const uint MESH_MATERIAL_FLAG_EMISSIVE_USED_AS_LIGHT = 1;

struct MeshMaterial {
    float base_color_mult[4];
    uint normal_map;
    uint spec_map;
    uint albedo_map;
    uint emissive_map;
    float roughness_mult;
    float metalness_factor;
    float emissive[3];
    uint flags;
    float map_transforms[6 * 4];
};

float2 transform_material_uv(MeshMaterial mat, float2 uv, uint map_idx) {
    uint xo = map_idx * 6;
    float2x2 rot_scl = float2x2(mat.map_transforms[xo+0], mat.map_transforms[xo+1], mat.map_transforms[xo+2], mat.map_transforms[xo+3]);
    float2 offset = float2(mat.map_transforms[xo+4], mat.map_transforms[xo+5]);
    return mul(rot_scl, uv) + offset;
}


#endif