#include <bit>
#include <vector>

#include "cascades.h"
#include "shared_types.h"


namespace {
    gpu::GpuCpuRange<CascadeData> allocate_cascade_data(const cascades::CascadesInfo& info) {
        auto allocated = info.allocate_frame_data(info.user_data, sizeof(CascadeData), 16);
        return {
            .cpu = reinterpret_cast<CascadeData*>(allocated.cpu),
            .gpu = reinterpret_cast<CascadeData*>(allocated.gpu),
            .size = allocated.size,
        };
    }
}

namespace cascades {
    struct Cascade {
        gpu::uint16 width;
        gpu::uint16 rwidth;
        gpu::uint16 cascade_cones;
        gpu::uint16 traces_texture_slot;
        gpu::uint16 cones_texture_slot;
    };

    struct Cascades {
        CascadesInfo info;
        std::vector<Cascade> cascades;
        gpu::PSO* compute_traces_pso;
        gpu::PSO* approximate_traces_pso;
        gpu::PSO* compute_cone_fluence_pso;
        gpu::PSO* accumulate_cone_fluence_pso;
    };

    Cascades* create_cascades(const CascadesInfo& info) {
        const gpu::uint32 n_cascades = std::bit_width(std::max(info.cascade0_size.x, info.cascade0_size.y)-1u);

        Cascades* cascades = new Cascades();
        cascades->info = info;
        cascades->cascades.resize(n_cascades);

        gpu::uint16 cascade_width = static_cast<gpu::uint16>(info.cascade0_size.x);
        gpu::uint16 cascade_rwidth = static_cast<gpu::uint16>(info.cascade0_size.y);
        gpu::uint16 texture_width = std::max(cascade_width, cascade_rwidth);
        gpu::uint16 texture_height = texture_width;
        gpu::uint16 cones_texture_width = texture_width;
        gpu::uint16 cascade_cones = 1;

        gpu::TextureDesc texture_desc {
            .extent = {.x = texture_width, .y = texture_height, .z = 1u},
            .format = gpu::Format::rgba16_float,
            .usage = gpu::TextureUsage::storage | gpu::TextureUsage::sampled
        };
        for (uint32 i=0; i<cascades->cascades.size(); ++i) {
            cascades->cascades[i].width = cascade_width;
            cascades->cascades[i].rwidth = cascade_rwidth;
            cascades->cascades[i].cascade_cones = cascade_cones;

            texture_desc.extent.x = texture_width*(cascade_cones+1u);
            cascades->cascades[i].traces_texture_slot = info.allocate_texture(info.user_data, texture_desc);

            cones_texture_width = std::max<gpu::uint16>(cones_texture_width, texture_width*cascade_cones);
            texture_width = (texture_width+1)/2;
            cascade_width = (cascade_width+1)/2;
            cascade_rwidth = (cascade_rwidth+1)/2;
            cascade_cones = cascade_cones*2;
        }

        texture_desc.format = gpu::Format::rg11b10_float;
        texture_desc.extent = {.x = info.cascade0_size.x, .y = info.cascade0_size.y, .z = 1};
        cascades->cascades[0].cones_texture_slot = info.allocate_texture(info.user_data, texture_desc);
        texture_desc.extent = {.x = cones_texture_width, .y = texture_height, .z = 1};
        for (gpu::uint32 i=0; i<2; ++i) {
            auto texture_slot = info.allocate_texture(info.user_data, texture_desc);
            for (gpu::uint32 j=i+1; j<n_cascades; j+=2) {
                cascades->cascades[j].cones_texture_slot = texture_slot;
            }
        }

        cascades->compute_traces_pso = gpu::create_compute_pso(info.device, {
            .code = info.load_shader(info.user_data, "compute_trace"),
            .entry_point = "compute_trace",
            .threadgroup_size = {.x = 16, .y = 16, .z = 1}
        });

        cascades->approximate_traces_pso = gpu::create_compute_pso(info.device, {
            .code = info.load_shader(info.user_data, "approximate_trace"),
            .entry_point = "approximate_trace",
            .threadgroup_size = {.x = 16, .y = 16, .z = 1}
        });

        cascades->compute_cone_fluence_pso = gpu::create_compute_pso(info.device, {
            .code = info.load_shader(info.user_data, "compute_cone_fluence"),
            .entry_point = "compute_cone_fluence",
            .threadgroup_size = {.x = 16, .y = 16, .z = 1}
        });

        cascades->accumulate_cone_fluence_pso = gpu::create_compute_pso(info.device, {
            .code = info.load_shader(info.user_data, "accumulate_cone_fluence"),
            .entry_point = "accumulate_cone_fluence",
            .threadgroup_size = {.x = 16, .y = 16, .z = 1}
        });

        return cascades;
    }

    gpu::uint16 compute_radiance(Cascades* cascades, gpu::uint16 world_texture_id, gpu::uint32x2 world_size, gpu::CommandBuffer* commands) {
        for (gpu::uint8 direction = 0; direction<4; ++direction) {
            const std::size_t run_cascades = std::bit_width((direction & 1 ? cascades->cascades[0].rwidth : cascades->cascades[0].width)-1u);
            constexpr int n_traced=3;
            for (std::size_t i=0; i<n_traced && i<run_cascades; ++i) {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate_cascade_data(cascades->info);
                cascade_data.cpu->texture_indices = (world_texture_id & 0xFF) | (cascades->cascades[i].traces_texture_slot << 8) | (direction << 24);
                cascade_data.cpu->cones_count = cascades->cascades[i].cascade_cones;
                cascade_data.cpu->width1 = direction & 1 ? cascades->cascades[i].rwidth : cascades->cascades[i].width;
                cascade_data.cpu->height1 = direction & 1 ? cascades->cascades[0].width : cascades->cascades[0].rwidth;
                cascade_data.cpu->width2 = static_cast<gpu::uint16>(direction & 1 ? world_size.y : world_size.x);
                cascade_data.cpu->height2 = static_cast<gpu::uint16>(direction & 1 ? world_size.x : world_size.y);
                bind_pso(commands, cascades->compute_traces_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascade_data.cpu->width1*(cascades->cascades[i].cascade_cones+1u)+15u)/16u, .y = (cascade_data.cpu->height1+15u)/16u, .z = 1});
            }
            barrier(commands,
                gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            for (std::size_t i=n_traced; i<run_cascades; ++i) {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate_cascade_data(cascades->info);
                cascade_data.cpu->texture_indices = cascades->cascades[i].traces_texture_slot | (cascades->cascades[i-1].traces_texture_slot << 8);
                cascade_data.cpu->cones_count = cascades->cascades[i].cascade_cones;
                cascade_data.cpu->width1 = direction & 1 ? cascades->cascades[i].rwidth : cascades->cascades[i].width;
                cascade_data.cpu->height1 = direction & 1 ? cascades->cascades[0].width : cascades->cascades[0].rwidth;
                cascade_data.cpu->width2 = direction & 1 ? cascades->cascades[i-1].rwidth : cascades->cascades[i-1].width;
                cascade_data.cpu->height2 = direction & 1 ? cascades->cascades[0].width : cascades->cascades[0].rwidth;
                bind_pso(commands, cascades->approximate_traces_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascade_data.cpu->width1*(cascades->cascades[i].cascade_cones+1u)+15u)/16u, .y = (cascade_data.cpu->height1+15u)/16u, .z = 1u});
                barrier(commands,
                    gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                    gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            }
            {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate_cascade_data(cascades->info);
                cascade_data.cpu->texture_indices = cascades->cascades[run_cascades-1].cones_texture_slot | (cascades->cascades[run_cascades-1].traces_texture_slot << 8);
                cascade_data.cpu->cones_count = cascades->cascades[run_cascades-1].cascade_cones;
                cascade_data.cpu->width1 = direction & 1 ? cascades->cascades[run_cascades-1].rwidth : cascades->cascades[run_cascades-1].width;
                cascade_data.cpu->height1 = direction & 1 ? cascades->cascades[0].width : cascades->cascades[0].rwidth;
                bind_pso(commands, cascades->compute_cone_fluence_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascade_data.cpu->width1*cascades->cascades[run_cascades-1].cascade_cones+15u)/16u, .y = (cascade_data.cpu->height1+15u)/16u, .z = 1u});
                barrier(commands,
                    gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                    gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            }
            for (auto i=run_cascades-1; i-->0;) {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate_cascade_data(cascades->info);
                cascade_data.cpu->texture_indices = cascades->cascades[i].cones_texture_slot | (cascades->cascades[i].traces_texture_slot << 8) | (cascades->cascades[i+1].cones_texture_slot << 16);
                cascade_data.cpu->cones_count = cascades->cascades[i].cascade_cones;
                cascade_data.cpu->width1 = direction & 1 ? cascades->cascades[i].rwidth : cascades->cascades[i].width;
                cascade_data.cpu->height1 = direction & 1 ? cascades->cascades[0].width : cascades->cascades[0].rwidth;
                cascade_data.cpu->width2 = direction & 1 ? cascades->cascades[i+1].rwidth : cascades->cascades[i+1].width;
                cascade_data.cpu->height2 = direction & 1 ? cascades->cascades[0].width : cascades->cascades[0].rwidth;
                if (i==0) {
                    cascade_data.cpu->texture_indices |= ((direction+1) << 24);
                }
                bind_pso(commands, cascades->accumulate_cone_fluence_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascade_data.cpu->width1*cascades->cascades[i].cascade_cones+15u)/16u, .y = (cascade_data.cpu->height1+15u)/16u, .z = 1u});
                barrier(commands,
                    gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                    gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            }
        }
        return cascades->cascades[0].cones_texture_slot;
    }

    void destroy_cascades(Cascades* cascades) {
        for (auto& cascade: cascades->cascades) {
            cascades->info.free_texture(cascades->info.user_data, cascade.traces_texture_slot);
        }
        for (std::size_t i=0; i<3; ++i) {
            cascades->info.free_texture(cascades->info.user_data, cascades->cascades[i].cones_texture_slot);
        }
        gpu::destroy_pso(cascades->compute_traces_pso);
        gpu::destroy_pso(cascades->approximate_traces_pso);
        gpu::destroy_pso(cascades->compute_cone_fluence_pso);
        gpu::destroy_pso(cascades->accumulate_cone_fluence_pso);

        delete(cascades);
    }
}
