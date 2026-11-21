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
    uint geometry_index = cascade_data.texture_indices & 0xFF;
    uint cascade_index = (cascade_data.texture_indices >> 8) & 0xFF;
    Texture2D<float4> geometry = gpu_texture<Texture2D<float4>>(geometry_index);
    Texture3D<float4> cascade0 = gpu_texture<Texture3D<float4>>(cascade_index);
    SamplerState nearest = gpu_sampler(0);

    int width, height, depth;
    cascade0.GetDimensions(width, height, depth);

    float4 sample_value = geometry.Sample(nearest, input.uv);

    int4 cascade0_coords = int4(int(input.uv.x*width)+1, int(input.uv.y*height), 0, 0);
    for (int i = 0; i < depth; ++i) {
        sample_value += 0.5*cascade0.Load(cascade0_coords) + 0.125*(
            cascade0.Load(cascade0_coords + int4(-1,0,0,0))+
            cascade0.Load(cascade0_coords + int4(1,0,0,0))+
            cascade0.Load(cascade0_coords + int4(0,-1,0,0))+
            cascade0.Load(cascade0_coords + int4(0,1,0,0)));
        cascade0_coords.z += 1;
    }
    return sample_value;
}

GPU_ROOT(CascadeData, cascade_data);

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_trace(int3 trace_coord: SV_DispatchThreadID) {
    uint geometry_index = cascade_data.texture_indices & 0xFF;
    uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    Texture2D<float4> geometry = gpu_texture<Texture2D<float4>>(geometry_index);
    RWTexture3D<float4> traces = gpu_texture<RWTexture3D<float4>>(traces_index);

    int traces_width, traces_height, traces_count;
    traces.GetDimensions(traces_width, traces_height, traces_count);
    if (trace_coord.x >= traces_width | trace_coord.y >= traces_height) {
        return;
    }
    const int cones_count = traces_count-1;
    float geometry_width, geometry_height;
    geometry.GetDimensions(geometry_width, geometry_height);

    float y_step = (2*trace_coord.z-cones_count)/float(cones_count);
    float geometry_dy = 0;  // TODO: try DDA
    for (int geometry_x = trace_coord.x*cones_count; geometry_x <= trace_coord.x*cones_count+cones_count; ++geometry_x) {
        float4 sample_value = geometry[int2(geometry_x, trace_coord.y+int(rint(geometry_dy)))];
        if (sample_value.a > 0.1) {
            traces[trace_coord] = sample_value;
            return;
        }
        geometry_dy += y_step;
    }

    traces[trace_coord] = float4(0.0);
}

float4 merge_traces(float4 near, float4 far) {
    return near + (1.0 - near.a) * far;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void approximate_trace(uint3 cascade_coord: SV_DispatchThreadID) {
    uint cascade_index = cascade_data.texture_indices & 0xFF;
    RWTexture3D<float4> cascade = gpu_texture<RWTexture3D<float4>>(cascade_index);
    RWTexture3D<float4> prev_cascade = gpu_texture<RWTexture3D<float4>>(cascade_index-1);

    int cascade_width, cascade_height, cascade_depth;
    cascade.GetDimensions(cascade_width, cascade_height, cascade_depth);
    if (cascade_coord.x >= cascade_width | cascade_coord.y >= cascade_height) {
        return;
    }

    if ((cascade_coord.z & 1) == 1) {
        cascade[cascade_coord] = 0.5 * (
            merge_traces(
                prev_cascade[uint3(cascade_coord.x*2, cascade_coord.y, cascade_coord.z/2)],
                prev_cascade[uint3(cascade_coord.x*2+1, cascade_coord.y+cascade_coord.z-1-cascade_depth/2, cascade_coord.z/2+1)]) +
            merge_traces(
                prev_cascade[uint3(cascade_coord.x*2, cascade_coord.y, cascade_coord.z/2+1)],
                prev_cascade[uint3(cascade_coord.x*2+1, cascade_coord.y+cascade_coord.z+1-cascade_depth/2, cascade_coord.z/2)]));
    } else {
        cascade[cascade_coord] = merge_traces(
            prev_cascade[uint3(cascade_coord.x*2, cascade_coord.y, cascade_coord.z/2)],
            prev_cascade[uint3(cascade_coord.x*2+1, cascade_coord.y+cascade_coord.z-cascade_depth/2, cascade_coord.z/2)]);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_cone_fluence(uint3 cone_coord: SV_DispatchThreadID) {
    uint cones_index = (cascade_data.texture_indices) & 0xFF;
    uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    RWTexture3D<float4> cones = gpu_texture<RWTexture3D<float4>>(cones_index);
    RWTexture3D<float4> traces = gpu_texture<RWTexture3D<float4>>(traces_index);

    int cones_width, cones_height, cones_count;
    cones.GetDimensions(cones_width, cones_height, cones_count);
    if (cone_coord.x >= cones_width | cone_coord.y >= cones_height) {
        return;
    }

    const int next_cone_idx = cone_coord.z*2;
    const float3 trace_angles = atan2(
        float3(next_cone_idx-cones_count, next_cone_idx+1-cones_count, next_cone_idx+2-cones_count),
        float3(cones_count));
    const float2 cone_angles = (trace_angles.yz-trace_angles.xy)*0.5/3.14159265;
    cones[cone_coord].rgb =
        cone_angles.x*traces[uint3(cone_coord.x, cone_coord.y, cone_coord.z)].rgb +
        cone_angles.y*traces[uint3(cone_coord.x, cone_coord.y, cone_coord.z+1)].rgb;
}

float3 merge_trace_cone(float4 near, float3 far) {
    return near.rgb + (1.0 - near.a) * far;
}

float4 scale_trace(float s, float4 trace) {
    return float4(s * trace.rgb, trace.a);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void accumulate_cone_fluence(uint3 cone_coord: SV_DispatchThreadID) {
    uint cones_index = (cascade_data.texture_indices) & 0xFF;
    uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    RWTexture3D<float4> cones = gpu_texture<RWTexture3D<float4>>(cones_index);
    RWTexture3D<float4> traces = gpu_texture<RWTexture3D<float4>>(traces_index);
    RWTexture3D<float4> next_cones = gpu_texture<RWTexture3D<float4>>(cones_index+1);

    int cones_width, cones_height, cones_count;
    cones.GetDimensions(cones_width, cones_height, cones_count);
    if (cone_coord.x >= cones_width | cone_coord.y >= cones_height) {
        return;
    }

    const int next_cone_idx = cone_coord.z*2;
    const float3 trace_angles = atan2(
        float3(next_cone_idx-cones_count, next_cone_idx+1-cones_count, next_cone_idx+2-cones_count),
        float3(cones_count));
    const float2 cone_angles = (trace_angles.yz-trace_angles.xy)*0.5/3.14159265;
    if ((cone_coord.x & 1) == 1) {
        cones[cone_coord].rgb =
            merge_trace_cone(
                scale_trace(cone_angles.x, traces[int3(cone_coord.x, cone_coord.y, cone_coord.z)]),
                next_cones[int3(cone_coord.x/2+1, cone_coord.y+next_cone_idx-cones_count, next_cone_idx)].rgb) +
            merge_trace_cone(
                scale_trace(cone_angles.y, traces[int3(cone_coord.x, cone_coord.y, cone_coord.z+1)]),
                next_cones[int3(cone_coord.x/2+1, cone_coord.y+next_cone_idx+2-cones_count, next_cone_idx+1)].rgb);
    } else {
        cones[cone_coord].rgb = 0.5*(
            next_cones[int3(cone_coord.x/2, cone_coord.y, next_cone_idx)].rgb +
            merge_trace_cone(
                scale_trace(cone_angles.x, merge_traces(
                    traces[int3(cone_coord.x, cone_coord.y, cone_coord.z)],
                    traces[int3(cone_coord.x+1, cone_coord.y+next_cone_idx-cones_count, cone_coord.z)])),
                next_cones[int3(cone_coord.x/2+1, cone_coord.y+2*(next_cone_idx-cones_count), next_cone_idx)].rgb) +

            next_cones[int3(cone_coord.x/2, cone_coord.y, next_cone_idx+1)].rgb +
            merge_trace_cone(
                scale_trace(cone_angles.y, merge_traces(
                    traces[int3(cone_coord.x, cone_coord.y, cone_coord.z+1)],
                    traces[int3(cone_coord.x+1, cone_coord.y+next_cone_idx+2-cones_count, cone_coord.z+1)])),
                next_cones[int3(cone_coord.x/2+1, cone_coord.y+2*(next_cone_idx+2-cones_count), next_cone_idx+1)].rgb));
    }
}
