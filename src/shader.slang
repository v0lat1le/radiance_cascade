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
    Texture2D<float3> cascade0 = gpu_texture<Texture2D<float3>>(cascade_index);
    SamplerState nearest = gpu_sampler(0);

    const float2 cascade0_extents = (cascade_data.width1, cascade_data.height1);
    const uint3 cascade0_coords = uint3(uint2(input.uv*cascade0_extents), 0);
    const float3 fluenence = 0.5*cascade0.Load(cascade0_coords) + 0.125*(
        cascade0.Load(cascade0_coords + int3(-1,0,0)) +
        cascade0.Load(cascade0_coords + int3(1,0,0)) +
        cascade0.Load(cascade0_coords + int3(0,-1,0)) +
        cascade0.Load(cascade0_coords + int3(0,1,0)));

    const float4 sample_value = geometry.Sample(nearest, input.uv);
    return float4(sample_value.rgb + (1.0-sample_value.a)*fluenence, sample_value.a);
}

GPU_ROOT(CascadeData, cascade_data);

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_trace(uint3 trace_coords: SV_DispatchThreadID) {
    const uint geometry_index = cascade_data.texture_indices & 0xFF;
    const uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    const uint direction = (cascade_data.texture_indices >> 24) & 0xFF;
    Texture2D<float4> geometry = gpu_texture<Texture2D<float4>>(geometry_index);
    RWTexture2D<float4> traces = gpu_texture<RWTexture2D<float4>>(traces_index);

    const uint cones_count = cascade_data.cones_count;
    const uint traces_count = cones_count + 1;
    const uint2 traces_extents = uint2(cascade_data.width1, cascade_data.height1);
    const uint2 geometry_extents = uint2(cascade_data.width2, cascade_data.height2);
    if (trace_coords.x >= traces_extents.x*traces_count | trace_coords.y >= traces_extents.y) {
        return;
    }

    const uint trace_idx = trace_coords.x/traces_extents.x;
    const uint trace_x = trace_coords.x%traces_extents.x;
    const uint geometry_coords_x = trace_x*cones_count;  // TODO: this assumes 1:1 R0 probe placement
    const float sample_step_y = (2.0*trace_idx-cones_count)/float(cones_count);  // TODO: try DDA
    uint2 geometry_coords;
    float2 sample_step;
    switch (direction) {
        case 0:
            geometry_coords = uint2(geometry_coords_x, trace_coords.y);
            sample_step = float2(1.0, sample_step_y);
        break;
        case 1:
            geometry_coords = uint2(geometry_extents.x-1-trace_coords.y, geometry_coords_x);
            sample_step = float2(-sample_step_y, 1.0);
        break;
        case 2:
            geometry_coords = geometry_extents-1 - uint2(geometry_coords_x, trace_coords.y);
            sample_step = float2(-1.0, -sample_step_y);
        break;
        case 3:
            geometry_coords = uint2(trace_coords.y, geometry_extents.y-1-geometry_coords_x);
            sample_step = float2(sample_step_y, -1.0);
        break;
    }
    for (int step = 0; step <= cones_count; ++step) {
        float4 sample_value = geometry[geometry_coords + int2(rint(sample_step*step))];
        if (sample_value.a > 0.1) {
            traces[trace_coords.xy] = sample_value;
            return;
        }
    }
    traces[trace_coords.xy] = float4(0.0);
}

float4 merge_traces(float4 near, float4 far) {
    return near + (1.0 - near.a)*far;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void approximate_trace(uint3 trace_coords: SV_DispatchThreadID) {
    const uint traces_index = cascade_data.texture_indices & 0xFF;
    const uint prev_traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    RWTexture2D<float4> traces = gpu_texture<RWTexture2D<float4>>(traces_index);
    Texture2D<float4> prev_traces = gpu_texture<Texture2D<float4>>(prev_traces_index);

    const uint cones_count = cascade_data.cones_count;
    const uint traces_count = cones_count + 1;
    const uint2 traces_extents = uint2(cascade_data.width1, cascade_data.height1);
    const uint2 prev_traces_extents = uint2(cascade_data.width2, cascade_data.height2);
    if (trace_coords.x >= traces_extents.x*traces_count | trace_coords.y >= traces_extents.y) {
        return;
    }

    const uint trace_idx = trace_coords.x/traces_extents.x;
    const uint trace_x = trace_coords.x%traces_extents.x;
    if ((trace_idx & 1) == 1) {
        traces[trace_coords.xy] = 0.5 * (
            merge_traces(
                prev_traces[uint2(trace_x*2+(trace_idx-1)*traces_extents.x, trace_coords.y)],
                prev_traces[uint2(trace_x*2+1+(trace_idx+1)*traces_extents.x, trace_coords.y+trace_idx-1-traces_count/2)]) +
            merge_traces(
                prev_traces[uint2(trace_x*2+(trace_idx+1)*traces_extents.x, trace_coords.y)],
                prev_traces[uint2(trace_x*2+1+(trace_idx-1)*traces_extents.x, trace_coords.y+trace_idx+1-traces_count/2)]));
    } else {
        traces[trace_coords.xy] = merge_traces(
            prev_traces[uint2(trace_x*2+trace_idx*traces_extents.x, trace_coords.y)],
            prev_traces[uint2(trace_x*2+1+trace_idx*traces_extents.x, trace_coords.y+trace_idx-traces_count/2)]);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_cone_fluence(uint3 cone_coords: SV_DispatchThreadID) {
    const uint cones_index = (cascade_data.texture_indices) & 0xFF;
    const uint traces_index = (cascade_data.texture_indices >> 8) & 0xFF;
    RWTexture2D<float3> cones = gpu_texture<RWTexture2D<float3>>(cones_index);
    Texture2D<float4> traces = gpu_texture<Texture2D<float4>>(traces_index);

    const uint cones_count = cascade_data.cones_count;
    const uint2 cones_extents = uint2(cascade_data.width1, cascade_data.height1);
    if (cone_coords.x >= cones_extents.x*cones_count | cone_coords.y >= cones_extents.y) {
        return;
    }

    const uint cone_idx = cone_coords.x/cones_extents.x;
    const uint cone_x = cone_coords.x%cones_extents.x;
    const uint next_cone_idx = cone_idx*2;
    const float3 trace_angles = atan2(
        float3(float(next_cone_idx)-cones_count, float(next_cone_idx)+1-cones_count, float(next_cone_idx)+2-cones_count),
        float3(cones_count));
    const float2 cone_angles = (trace_angles.yz-trace_angles.xy)*0.5/3.14159265;
    cones[cone_coords.xy] =
        cone_angles.x*traces[uint2(cone_x+cone_idx*cones_extents.x, cone_coords.y)].rgb +
        cone_angles.y*traces[uint2(cone_x+(cone_idx+1)*cones_extents.x, cone_coords.y)].rgb;
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
    const uint next_cones_index = (cascade_data.texture_indices >> 16) & 0xFF;
    const uint direction = (cascade_data.texture_indices >> 24) & 0xFF;
    RWTexture2D<float3> cones = gpu_texture<RWTexture2D<float3>>(cones_index);
    Texture2D<float4> traces = gpu_texture<Texture2D<float4>>(traces_index);
    Texture2D<float3> next_cones = gpu_texture<Texture2D<float3>>(next_cones_index);

    const uint cones_count = cascade_data.cones_count;
    const uint2 cones_extents = uint2(cascade_data.width1, cascade_data.height1);
    const uint2 next_cones_extents = uint2(cascade_data.width2, cascade_data.height2);
    if (cone_coords.x >= cones_extents.x*cones_count | cone_coords.y >= cones_extents.y) {
        return;
    }

    const uint cone_idx = cone_coords.x/cones_extents.x;
    const uint cone_x = cone_coords.x%cones_extents.x;
    const uint next_cone_idx = cone_idx*2;
    const float3 trace_angles = atan2(
        float3(float(next_cone_idx)-cones_count, float(next_cone_idx)+1-cones_count, float(next_cone_idx)+2-cones_count),
        float3(cones_count));
    const float2 cone_angles = (trace_angles.yz-trace_angles.xy)*0.5/3.14159265;
    float3 cone_fluence;

    if ((cone_x & 1) == 1) {
        if (cone_x/2+1 == cones_extents.x/2) {
            cone_fluence = 
                cone_angles.x*traces[uint2(cone_x+cone_idx*cones_extents.x, cone_coords.y)].rgb +
                cone_angles.y*traces[uint2(cone_x+(cone_idx+1)*cones_extents.x, cone_coords.y)].rgb;
        } else {
            cone_fluence =
                merge_trace_cone(
                    scale_trace(cone_angles.x, traces[uint2(cone_x+cone_idx*cones_extents.x, cone_coords.y)]),
                    next_cones[uint2(cone_x/2+1+next_cone_idx*cones_extents.x/2, cone_coords.y+next_cone_idx-cones_count)].rgb) +
                merge_trace_cone(
                    scale_trace(cone_angles.y, traces[uint2(cone_x+(cone_idx+1)*cones_extents.x, cone_coords.y)]),
                    next_cones[uint2(cone_x/2+1+(next_cone_idx+1)*cones_extents.x/2, cone_coords.y+next_cone_idx+2-cones_count)].rgb);
        }
    } else {
        if (cone_x/2+1 == cones_extents.x/2) {
            cone_fluence = 0.5*(
                next_cones[uint2(cone_x/2+next_cone_idx*cones_extents.x/2, cone_coords.y)].rgb +
                cone_angles.x*merge_traces(
                        traces[uint2(cone_x+cone_idx*cones_extents.x, cone_coords.y)],
                        traces[uint2(cone_x+1+cone_idx*cones_extents.x, cone_coords.y+next_cone_idx-cones_count)]).rgb +
                next_cones[uint2(cone_x/2+(next_cone_idx+1)*cones_extents.x/2, cone_coords.y)].rgb +
                cone_angles.y*merge_traces(
                        traces[uint2(cone_x+(cone_idx+1)*cones_extents.x, cone_coords.y)],
                        traces[uint2(cone_x+1+(cone_idx+1)*cones_extents.x, cone_coords.y+next_cone_idx+2-cones_count)]).rgb);
        } else {
            cone_fluence = 0.5*(
                next_cones[uint2(cone_x/2+next_cone_idx*cones_extents.x/2, cone_coords.y)].rgb +
                merge_trace_cone(
                    scale_trace(cone_angles.x, merge_traces(
                        traces[uint2(cone_x+cone_idx*cones_extents.x, cone_coords.y)],
                        traces[uint2(cone_x+1+cone_idx*cones_extents.x, cone_coords.y+next_cone_idx-cones_count)])),
                    next_cones[uint2(cone_x/2+1+next_cone_idx*cones_extents.x/2, cone_coords.y+2*(next_cone_idx-cones_count))].rgb) +

                next_cones[uint2(cone_x/2+(next_cone_idx+1)*cones_extents.x/2, cone_coords.y)].rgb +
                merge_trace_cone(
                    scale_trace(cone_angles.y, merge_traces(
                        traces[uint2(cone_x+(cone_idx+1)*cones_extents.x, cone_coords.y)],
                        traces[uint2(cone_x+1+(cone_idx+1)*cones_extents.x, cone_coords.y+next_cone_idx+2-cones_count)])),
                    next_cones[uint2(cone_x/2+1+(next_cone_idx+1)*cones_extents.x/2, cone_coords.y+2*(next_cone_idx+2-cones_count))].rgb));
        }
    }
    if (direction == 0) {  // TODO: should be a separate shader?
        cones[cone_coords.xy] = cone_fluence;
        return;
    }
    if (cone_x == 0) {
        return;
    }
    switch (direction) {
        case 1:
            if (cone_x == cones_extents.x-1) {
                cones[cone_coords.xy] = float3(0.0);
            }
            cones[uint2(cone_x-1, cone_coords.y)] = cone_fluence;
            return;
        case 2:
            cones[uint2(cones_extents.x-cone_coords.y-1, cone_x-1)] += cone_fluence;
            return;
        case 3:
            cones[uint2(cones_extents.x-cone_x, cones_extents.y-1-cone_coords.y)] += cone_fluence;
            return;
        case 4:
            cones[uint2(cone_coords.y, cones_extents.y-cone_coords.x)] += cone_fluence;
            return;
    }
}
