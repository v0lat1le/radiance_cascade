#include <NoGraphicsAPI/shader.slang>

#include "shared_types.h"


static const float2 positions[] = {
    float2(-1.0f, -3.0f),
    float2(3.0f, 1.0f),
    float2(-1.0f, 1.0f)
};

struct VertexOutput {
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
};

float rand(float2 st) {
    return fract(sin(dot(st.xy, float2(12.9898,78.233)))*43758.5453123);
}

[shader("vertex")]
VertexOutput vertexMain(uint vertex_index: SV_VertexID) {
    VertexOutput output;
    output.position = float4(positions[vertex_index], 0.0f, 1.0f);
    output.uv = positions[vertex_index]*0.5+0.5;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input): SV_Target0 {
    Texture2D<float4> geometry = gpu_texture<Texture2D<float4>>(0);
    Texture3D<float4> cascade0 = gpu_texture<Texture3D<float4>>(1);
    SamplerState nearest = gpu_sampler(0);
    SamplerState blinear = gpu_sampler(1);

    int width, height, ray_count;
    cascade0.GetDimensions(width, height, ray_count);

    float4 sample_value = geometry.Sample(nearest, input.uv);

    float ray_count_inv = 1.0 / float(ray_count);
    float uv_z = 0.5*ray_count_inv;
    for (int i = 0; i < ray_count; ++i) {
        sample_value += cascade0.Sample(blinear, float3(input.uv, uv_z));
        uv_z += ray_count_inv;
    }
    return sample_value*ray_count_inv;
}

GPU_ROOT(CascadeData, cascade_data);

float4 cast_ray(float2 probe_uv, float angle, uint geometry_index, uint prev_cascade_index, float prev_cascade_z) {
    SamplerState nearest = gpu_sampler(0);
    Texture2D<float4> geometry = gpu_texture<Texture2D<float4>>(geometry_index);

    const float2 ray_dir = float2(cos(angle), -sin(angle));
    const float2 march_step = ray_dir * cascade_data.ray_step;
    float2 sample_uv = probe_uv + cascade_data.ray_start * ray_dir;
    float2 ray_end = clamp(probe_uv + cascade_data.ray_end * ray_dir, float2(0), float2(1));
    float2 steps_per_d = (ray_end - sample_uv) / march_step;
    int steps = int(min(steps_per_d.x, steps_per_d.y));
    for (; steps > 0; steps--) {
        sample_uv += march_step;
        float4 sample_value = geometry.SampleLevel(nearest, sample_uv, 0);
        if (sample_value.a > 0.1) {
            return sample_value;
        }
    }

    if (prev_cascade_index == 255) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    SamplerState blinear = gpu_sampler(1);
    Texture3D<float4> prev_cascade = gpu_texture<Texture3D<float4>>(prev_cascade_index);
    return prev_cascade.SampleLevel(blinear, float3(probe_uv, prev_cascade_z), 0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeCascade(uint3 cascade_coord: SV_DispatchThreadID) {
    uint geometry_index = cascade_data.texture_indices & 0xFF;
    uint cascade_index = (cascade_data.texture_indices >> 8) & 0xFF;
    uint prev_cascade_index = (cascade_data.texture_indices >> 16) & 0xFF;
    RWTexture3D<float4> cascade = gpu_texture<RWTexture3D<float4>>(cascade_index);

    float width, height, ray_bundles;
    cascade.GetDimensions(width, height, ray_bundles);

    if (cascade_coord.x >= width | cascade_coord.y >= height) {
        return;
    }

    float4 result = float4(0.0);
    for (int ray_idx = cascade_coord.z * 4; ray_idx < cascade_coord.z * 4 + 4; ray_idx++) {
        const float angle = (float(ray_idx) + 0.5) * cascade_data.angle_delta;
        float2 probe_uv = (float2(cascade_coord.xy) + 0.5) / float2(width, height);
        float prev_cascade_z = (0.25*ray_idx + 0.125) / ray_bundles;
        result += cast_ray(probe_uv, angle, geometry_index, prev_cascade_index, prev_cascade_z);
    }
    cascade[cascade_coord] = 0.25 * result;
}
