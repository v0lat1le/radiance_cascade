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

[shader("vertex")]
VertexOutput main_vertex(uint vertex_index: SV_VertexID) {
    VertexOutput output;
    output.position = float4(positions[vertex_index], 0.0f, 1.0f);
    output.uv = positions[vertex_index]*0.5+0.5;
    return output;
}

[shader("fragment")]
float4 main_fragment(VertexOutput input): SV_Target0 {
    const uint geometry_index = cascade_data.texture_indices & 0xFF;
    const uint cascade_index = (cascade_data.texture_indices >> 8) & 0xFF;
    Texture2D<float4> geometry = gpu_texture<Texture2D<float4>>(geometry_index);
    Texture3D<float4> cascade0 = gpu_texture<Texture3D<float4>>(cascade_index);
    SamplerState nearest = gpu_sampler(0);

    int width, height, depth;
    cascade0.GetDimensions(width, height, depth);

    int4 cascade0_coords = int4(int2(input.uv*float2(width, height)), 0, 0);
    float4 fluenence = 0.5*cascade0.Load(cascade0_coords) + 0.125*(
        cascade0.Load(cascade0_coords + int4(-1,0,0,0)) +
        cascade0.Load(cascade0_coords + int4(1,0,0,0)) +
        cascade0.Load(cascade0_coords + int4(0,-1,0,0)) +
        cascade0.Load(cascade0_coords + int4(0,1,0,0)));

    float4 sample_value = geometry.Sample(nearest, input.uv);
    return sample_value + (1.0-sample_value.a)*fluenence;
}

GPU_ROOT(CascadeData, cascade_data);

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_trace(int3 trace_coords: SV_DispatchThreadID) {
    const uint geometry_index = cascade_data.texture_indices & 0xFF;
    const uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    const uint direction = (cascade_data.texture_indices >> 24) & 0xFF;
    Texture2D<float4> geometry = gpu_texture<Texture2D<float4>>(geometry_index);
    RWTexture3D<float4> traces = gpu_texture<RWTexture3D<float4>>(traces_index);

    int traces_width, traces_height, traces_count;
    traces.GetDimensions(traces_width, traces_height, traces_count);
    if (trace_coords.x >= traces_width | trace_coords.y >= traces_height) {
        return;
    }
    int geometry_width, geometry_height;
    geometry.GetDimensions(geometry_width, geometry_height);

    const int cones_count = traces_count-1;
    int2 geometry_coords = int2(trace_coords.x*cones_count, trace_coords.y);
    float2 sample_step = float2(1.0, (2*trace_coords.z-cones_count)/float(cones_count));  // TODO: try DDA
    for (int dir = 0; dir < direction; ++dir) {
        geometry_coords = int2(geometry_width-1-geometry_coords.y, geometry_coords.x);
        sample_step = float2(-sample_step.y, sample_step.x);
    }
    for (int step = 0; step <= cones_count; ++step) {
        float4 sample_value = geometry[geometry_coords + int2(rint(sample_step*step))];
        if (sample_value.a > 0.1) {
            traces[trace_coords] = sample_value;
            return;
        }
    }
    traces[trace_coords] = float4(0.0);
}

float4 merge_traces(float4 near, float4 far) {
    return near + (1.0 - near.a)*far;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void approximate_trace(uint3 trace_coords: SV_DispatchThreadID) {
    const uint traces_index = cascade_data.texture_indices & 0xFF;
    RWTexture3D<float4> traces = gpu_texture<RWTexture3D<float4>>(traces_index);
    RWTexture3D<float4> prev_traces = gpu_texture<RWTexture3D<float4>>(traces_index-1);

    int traces_width, traces_height, traces_count;
    traces.GetDimensions(traces_width, traces_height, traces_count);
    if (trace_coords.x >= traces_width | trace_coords.y >= traces_height) {
        return;
    }

    if ((trace_coords.z & 1) == 1) {
        traces[trace_coords] = 0.5 * (
            merge_traces(
                prev_traces[uint3(trace_coords.x*2, trace_coords.y, trace_coords.z/2)],
                prev_traces[uint3(trace_coords.x*2+1, trace_coords.y+trace_coords.z-1-traces_count/2, trace_coords.z/2+1)]) +
            merge_traces(
                prev_traces[uint3(trace_coords.x*2, trace_coords.y, trace_coords.z/2+1)],
                prev_traces[uint3(trace_coords.x*2+1, trace_coords.y+trace_coords.z+1-traces_count/2, trace_coords.z/2)]));
    } else {
        traces[trace_coords] = merge_traces(
            prev_traces[uint3(trace_coords.x*2, trace_coords.y, trace_coords.z/2)],
            prev_traces[uint3(trace_coords.x*2+1, trace_coords.y+trace_coords.z-traces_count/2, trace_coords.z/2)]);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_cone_fluence(uint3 cone_coords: SV_DispatchThreadID) {
    const uint cones_index = (cascade_data.texture_indices) & 0xFF;
    const uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    RWTexture3D<float4> cones = gpu_texture<RWTexture3D<float4>>(cones_index);
    RWTexture3D<float4> traces = gpu_texture<RWTexture3D<float4>>(traces_index);

    int cones_width, cones_height, cones_count;
    cones.GetDimensions(cones_width, cones_height, cones_count);
    if (cone_coords.x >= cones_width | cone_coords.y >= cones_height) {
        return;
    }

    const int next_cone_idx = cone_coords.z*2;
    const float3 trace_angles = atan2(
        float3(next_cone_idx-cones_count, next_cone_idx+1-cones_count, next_cone_idx+2-cones_count),
        float3(cones_count));
    const float2 cone_angles = (trace_angles.yz-trace_angles.xy)*0.5/3.14159265;
    cones[cone_coords].rgb =
        cone_angles.x*traces[uint3(cone_coords.x, cone_coords.y, cone_coords.z)].rgb +
        cone_angles.y*traces[uint3(cone_coords.x, cone_coords.y, cone_coords.z+1)].rgb;
}

float3 merge_trace_cone(float4 near, float3 far) {
    return near.rgb + (1.0 - near.a)*far;
}

float4 scale_trace(float scale, float4 trace) {
    return float4(scale*trace.rgb, trace.a);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void accumulate_cone_fluence(uint3 cone_coords: SV_DispatchThreadID) {
    const uint cones_index = (cascade_data.texture_indices) & 0xFF;
    const uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    const uint direction = (cascade_data.texture_indices >> 24) & 0xFF;
    RWTexture3D<float4> cones = gpu_texture<RWTexture3D<float4>>(cones_index);
    RWTexture3D<float4> traces = gpu_texture<RWTexture3D<float4>>(traces_index);
    RWTexture3D<float4> next_cones = gpu_texture<RWTexture3D<float4>>(cones_index+1);

    int cones_width, cones_height, cones_count;
    cones.GetDimensions(cones_width, cones_height, cones_count);
    if (cone_coords.x >= cones_width | cone_coords.y >= cones_height) {
        return;
    }

    const int next_cone_idx = cone_coords.z*2;
    const float3 trace_angles = atan2(
        float3(next_cone_idx-cones_count, next_cone_idx+1-cones_count, next_cone_idx+2-cones_count),
        float3(cones_count));
    const float2 cone_angles = (trace_angles.yz-trace_angles.xy)*0.5/3.14159265;
    float3 cone_fluence;
    if ((cone_coords.x & 1) == 1) {
        cone_fluence =
            merge_trace_cone(
                scale_trace(cone_angles.x, traces[int3(cone_coords.x, cone_coords.y, cone_coords.z)]),
                next_cones[int3(cone_coords.x/2+1, cone_coords.y+next_cone_idx-cones_count, next_cone_idx)].rgb) +
            merge_trace_cone(
                scale_trace(cone_angles.y, traces[int3(cone_coords.x, cone_coords.y, cone_coords.z+1)]),
                next_cones[int3(cone_coords.x/2+1, cone_coords.y+next_cone_idx+2-cones_count, next_cone_idx+1)].rgb);
    } else {
        cone_fluence = 0.5*(
            next_cones[int3(cone_coords.x/2, cone_coords.y, next_cone_idx)].rgb +
            merge_trace_cone(
                scale_trace(cone_angles.x, merge_traces(
                    traces[int3(cone_coords.x, cone_coords.y, cone_coords.z)],
                    traces[int3(cone_coords.x+1, cone_coords.y+next_cone_idx-cones_count, cone_coords.z)])),
                next_cones[int3(cone_coords.x/2+1, cone_coords.y+2*(next_cone_idx-cones_count), next_cone_idx)].rgb) +

            next_cones[int3(cone_coords.x/2, cone_coords.y, next_cone_idx+1)].rgb +
            merge_trace_cone(
                scale_trace(cone_angles.y, merge_traces(
                    traces[int3(cone_coords.x, cone_coords.y, cone_coords.z+1)],
                    traces[int3(cone_coords.x+1, cone_coords.y+next_cone_idx+2-cones_count, cone_coords.z+1)])),
                next_cones[int3(cone_coords.x/2+1, cone_coords.y+2*(next_cone_idx+2-cones_count), next_cone_idx+1)].rgb));
    }
    if (direction == 0) {
        cones[cone_coords].rgb = cone_fluence;
        return;
    }
    cone_coords.x -= 1;
    if (direction == 1) {
        cones[cone_coords].rgb = cone_fluence;
        return;
    }
    for (int dir = 1; dir < direction; ++dir) {
        cone_coords = int3(cones_height-cone_coords.y-1, cone_coords.x, cone_coords.z);
    }
    cones[cone_coords].rgb += cone_fluence;
}
