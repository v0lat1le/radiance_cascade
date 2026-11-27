#pragma once

#include "NoGraphicsAPI/NoGraphicsAPI.hpp"

namespace cascades {
    struct Cascades;

    struct CascadesInfo {
        void* user_data;
        gpu::Device* device;
        gpu::uint32x2 cascade0_size;
        gpu::ByteSpan(*load_shader)(void* user_data, const char* name);
        gpu::uint16(*allocate_texture)(void* user_data, const gpu::TextureDesc& texture_desc);
        gpu::GpuCpuRange<gpu::byte>(*allocate_frame_data)(void* user_data, gpu::uint64 size, gpu::uint64 alignment);
        void(*free_texture)(void* user_data, gpu::uint16 texture_id);
    };

    Cascades* create_cascades(const CascadesInfo& info);
    gpu::uint16 compute_radiance(Cascades* cascades, gpu::uint16 world_texture_id, gpu::uint32x2 world_size, gpu::CommandBuffer* commands);
    void destroy_cascades(Cascades* cascades);
}
