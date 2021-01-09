struct VertexPacked {
	float4 data0;
};

struct Mesh {
    uint vertex_core_offset;
    uint vertex_uv_offset;
    uint vertex_mat_offset;
    uint vertex_aux_offset;
    uint mat_data_offset;
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

struct MeshMaterial {
    float base_color_mult[4];
    uint normal_map;
    uint spec_map;
    uint albedo_map;
    float emissive[3];
};
