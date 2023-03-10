#include "../inc/math.hlsl"
#include "../inc/samplers.hlsl"
#include "../inc/mesh.hlsl"
#include "../inc/pack_unpack.hlsl"
#include "../inc/frame_constants.hlsl"
#include "../inc/bindless.hlsl"
#include "../inc/rt.hlsl"

[[vk::binding(0, 3)]] RaytracingAccelerationStructure acceleration_structure;

struct RayHitAttrib {
    float2 bary;
};

float twice_triangle_area(float3 p0, float3 p1, float3 p2) {
    return length(cross(p1 - p0, p2 - p0));
}

float twice_uv_area(float2 t0, float2 t1, float2 t2) {
    return abs((t1.x - t0.x) * (t2.y - t0.y) - (t2.x - t0.x) * (t1.y - t0.y));
}

struct BindlessTextureWithLod {
    Texture2D tex;
    float lod;
};

// https://media.contentapi.ea.com/content/dam/ea/seed/presentations/2019-ray-tracing-gems-chapter-20-akenine-moller-et-al.pdf
BindlessTextureWithLod compute_texture_lod(uint bindless_texture_idx, float triangle_constant, float3 ray_direction, float3 surf_normal, float cone_width) {
    // Not using `GetDimensions` as it's buggy on AMD.
    float2 wh = bindless_texture_sizes[bindless_texture_idx].xy;

    float lambda = triangle_constant;
    lambda += log2(abs(cone_width));
    lambda += 0.5 * log2(wh.x * wh.y);

    // TODO: This blurs a lot at grazing angles; do aniso.
    lambda -= log2(abs(dot(normalize(ray_direction), surf_normal)));

    BindlessTextureWithLod res;
    res.tex = bindless_textures[NonUniformResourceIndex(bindless_texture_idx)];
    res.lod = lambda;
    return res;
}

[shader("closesthit")]
void main(inout GbufferRayPayload payload: SV_RayPayload, in RayHitAttrib attrib: SV_IntersectionAttributes) {
    float3 hit_point = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const float hit_dist = length(hit_point - WorldRayOrigin());

    float3 barycentrics = float3(1.0 - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

    //Mesh mesh = meshes[InstanceIndex() / 2];
    Mesh mesh = meshes[InstanceID()];

    // Indices of the triangle
    uint3 ind = uint3(
        vertices.Load((PrimitiveIndex() * 3 + 0) * sizeof(uint) + mesh.index_offset),
        vertices.Load((PrimitiveIndex() * 3 + 1) * sizeof(uint) + mesh.index_offset),
        vertices.Load((PrimitiveIndex() * 3 + 2) * sizeof(uint) + mesh.index_offset)
    );

    Vertex v0 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.x * sizeof(float4) + mesh.vertex_core_offset))));
    Vertex v1 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.y * sizeof(float4) + mesh.vertex_core_offset))));
    Vertex v2 = unpack_vertex(VertexPacked(asfloat(vertices.Load4(ind.z * sizeof(float4) + mesh.vertex_core_offset))));
    float3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;

    const float3 surf_normal_os = normalize(cross(v1.position - v0.position, v2.position - v0.position));
    const float3 surf_normal_ws = normalize(mul(ObjectToWorld3x4(), float4(surf_normal_os, 0.0)));

    if (frame_constants.render_overrides.has_flag(RenderOverrideFlags::FORCE_FACE_NORMALS)) {
        normal = surf_normal_os;
    }

    float4 v_color = 1.0.xxxx;
    if (mesh.vertex_aux_offset != 0) {
        float4 vc0 = asfloat(vertices.Load4(ind.x * sizeof(float4) + mesh.vertex_aux_offset));
        float4 vc1 = asfloat(vertices.Load4(ind.y * sizeof(float4) + mesh.vertex_aux_offset));
        float4 vc2 = asfloat(vertices.Load4(ind.z * sizeof(float4) + mesh.vertex_aux_offset));
        v_color = vc0 * barycentrics.x + vc1 * barycentrics.y + vc2 * barycentrics.z;
    }

    float2 uv0 = asfloat(vertices.Load2(ind.x * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv1 = asfloat(vertices.Load2(ind.y * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv2 = asfloat(vertices.Load2(ind.z * sizeof(float2) + mesh.vertex_uv_offset));
    float2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    const float cone_width = payload.ray_cone.width_at_t(hit_dist);
    const float3 v0_pos_ws = mul(ObjectToWorld3x4(), float4(v0.position, 1.0));
    const float3 v1_pos_ws = mul(ObjectToWorld3x4(), float4(v1.position, 1.0));
    const float3 v2_pos_ws = mul(ObjectToWorld3x4(), float4(v2.position, 1.0));
    const float lod_triangle_constant = 0.5 * log2(twice_uv_area(uv0, uv1, uv2) / twice_triangle_area(v0_pos_ws, v1_pos_ws, v2_pos_ws));

    uint material_id = vertices.Load(ind.x * sizeof(uint) + mesh.vertex_mat_offset);
    MeshMaterial material = vertices.Load<MeshMaterial>(mesh.mat_data_offset + material_id * sizeof(MeshMaterial));

    float2 albedo_uv = transform_material_uv(material, uv, 0);
    const BindlessTextureWithLod albedo_tex =
        compute_texture_lod(material.albedo_map, lod_triangle_constant, WorldRayDirection(), surf_normal_ws, cone_width);

    float3 albedo =
        albedo_tex.tex.SampleLevel(sampler_llr, albedo_uv, albedo_tex.lod).xyz
        * float4(material.base_color_mult).xyz
        * v_color.rgb;

    float2 spec_uv = transform_material_uv(material, uv, 2);
    const BindlessTextureWithLod spec_tex =
        compute_texture_lod(material.spec_map, lod_triangle_constant, WorldRayDirection(), surf_normal_ws, cone_width);
    float4 metalness_roughness = spec_tex.tex.SampleLevel(sampler_llr, spec_uv, spec_tex.lod);
    float perceptual_roughness = material.roughness_mult * metalness_roughness.x;
    float roughness = clamp(perceptual_roughness_to_roughness(perceptual_roughness), 1e-4, 1.0);
    float metalness = metalness_roughness.y * material.metalness_factor;

    if (frame_constants.render_overrides.has_flag(RenderOverrideFlags::NO_METAL)) {
        metalness = 0;
    }

    if (frame_constants.render_overrides.material_roughness_scale <= 1) {
        roughness *= frame_constants.render_overrides.material_roughness_scale;
    } else {
        roughness = square(lerp(sqrt(roughness), 1.0, 1.0 - 1.0 / frame_constants.render_overrides.material_roughness_scale));
    }

#if 0
    if (!frame_constants.render_overrides.has_flag(RenderOverrideFlags::NO_NORMAL_MAPS)) {
        float4 v_tangent_packed0 =
            select(mesh.vertex_tangent_offset != 0
                , asfloat(vertices.Load4(ind.x * sizeof(float4) + mesh.vertex_tangent_offset))
                , float4(1, 0, 0, 1));
        float4 v_tangent_packed1 =
            select(mesh.vertex_tangent_offset != 0
                , asfloat(vertices.Load4(ind.y * sizeof(float4) + mesh.vertex_tangent_offset))
                , float4(1, 0, 0, 1));
        float4 v_tangent_packed2 =
            select(mesh.vertex_tangent_offset != 0
                , asfloat(vertices.Load4(ind.z * sizeof(float4) + mesh.vertex_tangent_offset))
                , float4(1, 0, 0, 1));

        float3 tangent0 = v_tangent_packed0.xyz;
        float3 bitangent0 = normalize(cross(v0.normal, tangent0) * v_tangent_packed0.w);

        float3 tangent1 = v_tangent_packed1.xyz;
        float3 bitangent1 = normalize(cross(v1.normal, tangent1) * v_tangent_packed1.w);

        float3 tangent2 = v_tangent_packed2.xyz;
        float3 bitangent2 = normalize(cross(v2.normal, tangent2) * v_tangent_packed2.w);

        float3 tangent = tangent0 * barycentrics.x + tangent1 * barycentrics.y + tangent2 * barycentrics.z;
        float3 bitangent = bitangent0 * barycentrics.x + bitangent1 * barycentrics.y + bitangent2 * barycentrics.z;

        float2 normal_uv = transform_material_uv(material, uv, 0);
        const BindlessTextureWithLod normal_tex =
            compute_texture_lod(material.normal_map, lod_triangle_constant, WorldRayDirection(), surf_normal_ws, cone_width);

        float3 ts_normal = normal_tex.tex.SampleLevel(sampler_llr, normal_uv, normal_tex.lod).xyz * TODO;

        if (frame_constants.render_overrides.has_flag(RenderOverrideFlags::FLIP_NORMAL_MAP_YZ)) {
            ts_normal.zy *= -1;
        }

        if (dot(bitangent, bitangent) > 0.0) {
            float3x3 tbn = float3x3(tangent, bitangent, normal);
            normal = mul(ts_normal, tbn);
        }
        normal = normalize(normal);
    }
#endif

    float2 emissive_uv = transform_material_uv(material, uv, 3);
    const BindlessTextureWithLod emissive_tex =
        compute_texture_lod(material.emissive_map, lod_triangle_constant, WorldRayDirection(), surf_normal_ws, cone_width);

    float3 emissive = 0;

    // Only allow emissive if this is not a light
    // ... except then still allow it if the path is currently tracing from the eye,
    // since we need the direct contribution of the light's surface to the screen.
    if (0 == payload.path_length || 0 == (material.flags & MESH_MATERIAL_FLAG_EMISSIVE_USED_AS_LIGHT)) {
        emissive = 1.0.xxx
            * emissive_tex.tex.SampleLevel(sampler_llr, emissive_uv, emissive_tex.lod).rgb
            * float3(material.emissive)
            * instance_dynamic_parameters_dyn[InstanceIndex()].emissive_multiplier
            * frame_constants.pre_exposure;
    }

    GbufferData gbuffer = GbufferData::create_zero();
    gbuffer.albedo = albedo;
    gbuffer.normal = normalize(mul(ObjectToWorld3x4(), float4(normal, 0.0)));
    gbuffer.roughness = roughness;
    gbuffer.metalness = metalness;
    gbuffer.emissive = emissive;

    // Force double-sided
    if (dot(WorldRayDirection(), gbuffer.normal) > 0) {
        gbuffer.normal *= -1;
    }

    //gbuffer.albedo = float3(0.966653, 0.802156, 0.323968); // Au from Mitsuba

    payload.gbuffer_packed = gbuffer.pack();
    payload.t = RayTCurrent();
}
