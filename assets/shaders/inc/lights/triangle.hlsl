struct Triangle {
    float3 v;
    float3 e0;
    float3 e1;
};

struct TriangleLight {
    float packed[12];

    static TriangleLight from_packed(TriangleLightPacked p) {
        TriangleLight res;
        res.packed = p.packed;
        return res;
    }

    float3 vertex(uint i) {
        return float3(packed[i * 3 + 0], packed[i * 3 + 1], packed[i * 3 + 2]);
    }

    float3 e0() {
        return vertex(1) - vertex(0);
    }

    float3 e1() {
        return vertex(2) - vertex(0);
    }

    Triangle as_triangle() {
        Triangle res;
        res.v = vertex(0);
        res.e0 = e0();
        res.e1 = e1();
        return res;
    }

    float3 radiance() {
        return float3(packed[9], packed[10], packed[11]);
    }
};

float3 sample_point_on_triangle(Triangle tri, float2 urand) {
    float su0 = sqrt(urand.x);
    float b0 = 1.0 - su0;
    float b1 = urand.y * su0;

    return tri.v + b0 * tri.e0 + b1 * tri.e1;
}

// Solid angle measure
struct PdfSolAng {
    float value;
};

struct PdfArea {
    float value;
};

float to_projected_solid_angle_measure(PdfArea pdf, float ndotl, float lndotl, float sqdist) {
    return pdf.value * sqdist / ndotl / lndotl;
}

float to_projected_solid_angle_measure(PdfSolAng pdf, float ndotl, float lndotl, float sqdist) {
    return pdf.value / ndotl;
}

struct LightSampleResultArea {
    float3 pos;
    float3 normal;
    PdfArea pdf;
};

struct LightSampleResultSolAng {
    float3 pos;
    float3 normal;
    PdfSolAng pdf;
};

LightSampleResultArea sample_triangle_light(Triangle tri, float2 urand) {
    float3 perp = cross(tri.e0, tri.e1);
    float perp_inv_len = rsqrt(dot(perp, perp));

    LightSampleResultArea res;
    res.pos = sample_point_on_triangle(tri, urand);
    res.normal = perp * perp_inv_len;
    res.pdf.value = 2.0 * perp_inv_len;   // 1.0 / triangle area
    return res;
}

float3 sample_point_on_triangle_basu_owen(Triangle tri, float u) {
    float2 A = float2(1, 0);
    float2 B = float2(0, 1);
    float2 C = float2(0, 0);

    uint uf = uint(u * 4294967295.0);           // Convert to fixed point

    for (int i = 0; i < 16; ++i) {            // For each base-4 digit
        uint d = (uf >> (2 * (15 - i))) & 0x3; // Get the digit

        float2 An, Bn, Cn;
        switch (d) {
        case 0:
            An = (B + C) / 2;
            Bn = (A + C) / 2;
            Cn = (A + B) / 2;
            break;
        case 1:
            An = A;
            Bn = (A + B) / 2;
            Cn = (A + C) / 2;
            break;
        case 2:
            An = (B + A) / 2;
            Bn = B;
            Cn = (B + C) / 2;
            break;
        case 3:
            An = (C + A) / 2;
            Bn = (C + B) / 2;
            Cn = C;
            break;
        }
        A = An;
        B = Bn;
        C = Cn;
    }

    float2 r = (A + B + C) / 3.0;
    return tri.v + tri.e0 * r.x + tri.e1 * r.y;
}

LightSampleResultArea sample_triangle_light_basu_owen(Triangle tri, float urand) {
    float3 perp = cross(tri.e0, tri.e1);
    float perp_inv_len = 1.0 / sqrt(dot(perp, perp));

    LightSampleResultArea res;
    res.pos = sample_point_on_triangle_basu_owen(tri, urand);
    res.normal = perp * perp_inv_len;
    res.pdf.value = 2.0 * perp_inv_len;   // 1.0 / triangle area
    return res;
}

float3 intersect_ray_plane(float3 normal, float3 plane_pt, float3 o, float3 dir) {
    return o - dir * (dot(o - plane_pt, normal) / dot(dir, normal));
}

// Based on "The Solid Angle of a Plane Triangle" by Oosterom and Strackee
float spherical_triangle_area(float3 a, float3 b, float3 c) {
    float numer = abs(dot(a, cross(b, c)));
    float denom = 1.0 + dot(a, b) + dot(a, c) + dot(b, c);
    return atan2(numer, denom) * 2.0;
}

// Based on "Sampling for Triangular Luminaire", Graphics Gems III p312
// https://github.com/erich666/GraphicsGems/blob/master/gemsiii/luminaire/triangle_luminaire.C
LightSampleResultSolAng sample_light_sol_ang(float3 pt, Triangle tri, float2 urand) {
    float3 normal = normalize(cross(tri.e0, tri.e1));

    float3 p1_sph = normalize(tri.v - pt);
    float3 p2_sph = normalize(tri.v + tri.e0 - pt);
    float3 p3_sph = normalize(tri.v + tri.e1 - pt);

    float2 uv = float2(1.0 - sqrt(1.0 - urand.x), urand.y * sqrt(1.0 - urand.x));
    float3 x_sph = p1_sph + uv.x * (p2_sph - p1_sph) + uv.y * (p3_sph - p1_sph);

    float sample_sqdist = dot(x_sph, x_sph);

    float3 x_ = intersect_ray_plane(normal, tri.v, pt, x_sph);

    float3 proj_tri_norm = cross(p2_sph - p1_sph, p3_sph - p1_sph);
    float area = 0.5 * sqrt(dot(proj_tri_norm, proj_tri_norm));
    proj_tri_norm /= sqrt(dot(proj_tri_norm, proj_tri_norm));

    float l2ndotl = -dot(proj_tri_norm, x_sph) / sqrt(sample_sqdist);

    LightSampleResultSolAng res;
    res.pos = x_;
    res.normal = normal;
    res.pdf.value = sample_sqdist / (l2ndotl * area);

    return res;
}
