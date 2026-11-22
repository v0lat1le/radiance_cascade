#include "RGFW.h"
#include "imgui_impl_rgfw.h"
#include "imgui_impl_nographicsapi.h"

#include "NoGraphicsAPI/NoGraphicsAPI.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

#include "shared_types.h"


#define SCALE 1
#define N_CASCADES 10
#define DOT_SIZE 10

namespace {
#if defined(__APPLE__)
    using id = void*;
    using SEL = void*;
    extern "C" id sel_getUid(const char*);
    extern "C" id objc_msgSend();
    extern "C" SEL sel_registerName(const char*);

    id get_metal_layer(RGFW_window* window) {
        void* view = RGFW_window_getView_OSX(window);
        return ((id(*)(id, SEL))objc_msgSend)(view, sel_getUid("layer"));
    }
    void* init_surface(RGFW_window* window) {
        RGFW_window_setLayer_OSX(window, RGFW_getLayer_OSX());
        return get_metal_layer(window);
    }

    void update_surface(RGFW_window* window) {
        struct CGSize {
            double width;
            double height;
        };
        id layer = get_metal_layer(window);
        ((void (*)(id, SEL, double))objc_msgSend)(layer, sel_registerName("setContentsScale:"), RGFW_window_getMonitor(window)->pixelRatio);

        i32 pixel_width, pixel_height;
        RGFW_window_getSizeInPixels(window, &pixel_width, &pixel_height);
        ((void (*)(id, SEL, CGSize))objc_msgSend)(layer, sel_registerName("setDrawableSize:"), CGSize{static_cast<double>(pixel_width), static_cast<double>(pixel_height)});
    }
#else
    void* init_surface(RGFW_window* window) {
        return RGFW_window_getHWND(window);
    }

    void update_surface(RGFW_window* /*window*/) {}
#endif

    gpu::uint64 align_up(gpu::uint64 value, gpu::uint64 alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    template<typename T>
    gpu::GpuCpuRange<T> allocate(gpu::GpuCpuRange<gpu::byte>& heap, gpu::uint64 count=1, gpu::uint64 alignment=1, gpu::uint64 size_alignment=1) {
        gpu::uint64 size = align_up(sizeof(T)*count, size_alignment);
        gpu::uint64 align_pad = align_up(reinterpret_cast<gpu::uint64>(heap.gpu), alignment) - reinterpret_cast<gpu::uint64>(heap.gpu);
        assert(size + align_pad < heap.size);
        gpu::GpuCpuRange<T> result {
            .cpu = reinterpret_cast<T*>(heap.cpu+align_pad),
            .gpu = reinterpret_cast<T*>(heap.gpu+align_pad),
            .size = count*sizeof(T)
        };
        heap.cpu += align_pad+size;
        heap.gpu += align_pad+size;
        heap.size -= align_pad+size;
        return result;
    }

    std::vector<gpu::byte> read_file(const std::filesystem::path& path) {
        auto size = std::filesystem::file_size(path);
        std::ifstream file(path, std::ios::binary);
        std::vector<gpu::byte> result(size);
        file.read(reinterpret_cast<char*>(result.data()), size);
        return result;
    }

    void draw_dot(gpu::uint32x2 pos, gpu::uint32 color, gpu::Texture* tex, gpu::CommandBuffer* commands, gpu::GpuCpuRange<byte>& buffer) {
        constexpr gpu::uint32 dot_size = DOT_SIZE;
        gpu::GpuCpuRange<gpu::uint32> staging = allocate<gpu::uint32>(buffer, dot_size*dot_size);
        int half_size = dot_size/2;
        for (int i=0; i<half_size; ++i) {
            for (int j=0; j<half_size; ++j) {
                staging.cpu[(half_size+j)*dot_size+half_size-i-1] = 
                staging.cpu[(half_size-j-1)*dot_size+half_size-i-1] =
                staging.cpu[(half_size+j)*dot_size+half_size+i] = 
                staging.cpu[(half_size-j-1)*dot_size+half_size+i] =
                4*(i*i+j*j) < dot_size*dot_size ? color : 0;
            }
        }
        if (dot_size & 1) {
            for (int i=0; i<half_size+1; ++i) {
                staging.cpu[(half_size+i)*dot_size+half_size-i] = 
                staging.cpu[(half_size-i)*dot_size+half_size-i] =
                staging.cpu[(half_size+i)*dot_size+half_size+i] = 
                staging.cpu[(half_size-i)*dot_size+half_size+i] = color;
            }
        }

        gpu::copy_memory_to_texture(commands, gpu::gpu_range(staging), tex, {
            .offset = {.x = pos.x/SCALE-dot_size/2, .y = pos.y/SCALE-dot_size/2},
            .extent = {.x = dot_size, .y = dot_size, .z = 1}
        });
    }
}

int main() {
    RGFW_init("radiance-cascade", 0);
    RGFW_window* window = RGFW_createWindow("radiance-cascade", 0, 0, 1024, 1024, RGFW_windowCenter);

    gpu::Device* device = gpu::create_device({ .window = init_surface(window), .swapchain_format = gpu::Format::bgra8_srgb }).device;

    gpu::TimelinePoint latest_completion{ .semaphore = gpu::create_timeline_semaphore(device) };
    gpu::CommandPool* command_pools[] = { gpu::create_command_pool(device), gpu::create_command_pool(device) };
    gpu::GpuHeap heap = gpu::create_gpu_heap(device, 32*1024*1024);

    gpu::TextureHeap texture_heap = gpu::create_texture_heap(device, 1024*1024*1024);
    gpu::uint64 texture_heap_offset = 0;
    gpu::TextureDescriptorHeap* texture_descriptor_heap = gpu::create_texture_descriptor_heap(device, 32);
    gpu::SamplerDescriptorHeap* sampler_descriptor_heap = gpu::create_sampler_descriptor_heap(device, 2);

    gpu::TextureDesc texture_desc{
        .extent={.x = 1024/SCALE, .y = 1024/SCALE, .z = 1},
        .format = gpu::Format::bgra8_unorm,
        .usage = gpu::TextureUsage::sampled | gpu::TextureUsage::transfer_destination
    };
    gpu::SizeAlign texture_requirements = gpu::get_texture_size_align(device, texture_desc);
    texture_heap_offset = align_up(texture_heap_offset, texture_requirements.align);
    gpu::Texture* geometry = gpu::create_texture(device, texture_desc, texture_heap, texture_heap_offset);
    texture_heap_offset += texture_requirements.size;

    struct Cascade {
        gpu::uint32 width;
        gpu::uint32 height;
        gpu::uint32 cascade_cones;
        gpu::Texture* traces_texture;
        gpu::Texture* cones_texture;
        gpu::uint8 traces_texture_slot;
        gpu::uint8 cones_texture_slot;
    };
    std::vector<Cascade> cascades(N_CASCADES);

    uint32 cascade_width = texture_desc.extent.x;
    uint32 cascade_height = texture_desc.extent.y;
    uint32 cascade_cones = 1;

    texture_desc.type = gpu::TextureType::three_d;
    texture_desc.format = gpu::Format::rgba16_float;
    texture_desc.usage = gpu::TextureUsage::storage | gpu::TextureUsage::sampled;
    for (uint32 i=0; i<cascades.size(); ++i) {
        cascades[i].height = cascade_height;
        cascades[i].width = cascade_width;
        cascades[i].cascade_cones = cascade_cones;

        texture_desc.extent = {.x = cascade_width, .y = cascade_height, .z = cascade_cones+1};
        texture_requirements = gpu::get_texture_size_align(device, texture_desc);
        texture_heap_offset = align_up(texture_heap_offset, texture_requirements.align);
        cascades[i].traces_texture = gpu::create_texture(device, texture_desc, texture_heap, texture_heap_offset);
        texture_heap_offset += texture_requirements.size;

        texture_desc.extent = {.x = cascade_width, .y = cascade_height, .z = cascade_cones};
        texture_requirements = gpu::get_texture_size_align(device, texture_desc);
        texture_heap_offset = align_up(texture_heap_offset, texture_requirements.align);
        cascades[i].cones_texture = gpu::create_texture(device, texture_desc, texture_heap, texture_heap_offset);
        texture_heap_offset += texture_requirements.size;

        cascade_width = (cascade_width+1)/2;
        cascade_cones = cascade_cones*2;
    }

    {
        gpu::CommandBuffer* texture_commands = gpu::begin_commands(command_pools[0]);
        gpu::initialize_texture(texture_commands, geometry);
        gpu::GpuCpuRange<gpu::byte> init_heap = heap.range;
        gpu::GpuCpuRange<gpu::uint32> zeros = allocate<gpu::uint32>(init_heap, cascades[0].width*cascades[0].height);
        std::memset(zeros.cpu, 0, zeros.size);
        gpu::copy_memory_to_texture(texture_commands, gpu::gpu_range(zeros), geometry);
        for (uint32 i=0; i<cascades.size(); ++i) {
            gpu::initialize_texture(texture_commands, cascades[i].traces_texture);
            gpu::initialize_texture(texture_commands, cascades[i].cones_texture);
        }
        gpu::end_commands(texture_commands);
        latest_completion.value++;
        gpu::submit(device, {.commands = {texture_commands}, .completion = latest_completion});
    }

    auto shader = read_file("shaders/shader" SHADER_EXTENSION);
    gpu::PSO* render_cascades_pso = create_graphics_pso(device, {
        .vertex = {.code = {shader.data(), shader.size()}, .entry_point = "main_vertex"},
        .fragment = {.code = {shader.data(), shader.size()}, .entry_point = "main_fragment"},
        .color_targets = { { .format = gpu::Format::bgra8_srgb } }
    });

    gpu::PSO* compute_trace_pso = gpu::create_compute_pso(device, {
        .code = {shader.data(), shader.size()},
        .entry_point = "compute_trace",
        .threadgroup_size = {.x = 16, .y = 16, .z = 1}
    });

    gpu::PSO* approximate_traces_pso = gpu::create_compute_pso(device, {
        .code = {shader.data(), shader.size()},
        .entry_point = "approximate_trace",
        .threadgroup_size = {.x = 16, .y = 16, .z = 1}
    });

    gpu::PSO* compute_cone_fluence_pso = gpu::create_compute_pso(device, {
        .code = {shader.data(), shader.size()},
        .entry_point = "compute_cone_fluence",
        .threadgroup_size = {.x = 16, .y = 16, .z = 1}
    });

    gpu::PSO* accumulate_cone_fluence_pso = gpu::create_compute_pso(device, {
        .code = {shader.data(), shader.size()},
        .entry_point = "accumulate_cone_fluence",
        .threadgroup_size = {.x = 16, .y = 16, .z = 1}
    });

    gpu::write_sampler_descriptor(sampler_descriptor_heap, 0, {
        .min_filter = gpu::Filter::nearest,
        .mag_filter = gpu::Filter::nearest,
        .address_u = gpu::AddressMode::clamp_to_edge,
        .address_v = gpu::AddressMode::clamp_to_edge,
    });
    gpu::write_sampler_descriptor(sampler_descriptor_heap, 1, {
        .min_filter = gpu::Filter::linear,
        .mag_filter = gpu::Filter::linear,
        .address_u = gpu::AddressMode::clamp_to_edge,
        .address_v = gpu::AddressMode::clamp_to_edge,
    });
    gpu::write_texture_descriptor(texture_descriptor_heap, 0, geometry, gpu::TextureDescriptorType::sampled);
    for (uint8 i=0; i<cascades.size(); ++i) {
        cascades[i].traces_texture_slot = i+1;
        cascades[i].cones_texture_slot = static_cast<gpu::uint8>(i+1+cascades.size());
        gpu::write_texture_descriptor(texture_descriptor_heap, cascades[i].traces_texture_slot, cascades[i].traces_texture, gpu::TextureDescriptorType::storage);
        gpu::write_texture_descriptor(texture_descriptor_heap, cascades[i].cones_texture_slot, cascades[i].cones_texture, gpu::TextureDescriptorType::storage);
    }

    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_IsSRGB;

    ImGui::StyleColorsDark();
    // convert_imgui_style_to_linear_HACK();  // TODO: do this in the shader?

    ImGui_ImplRgfw_InitForOther(window, true);

    struct BackendUserData {
        gpu::Device* device;
        gpu::CommandBuffer* commands;
        gpu::GpuCpuRange<byte> frame_heap;
        gpu::TextureHeap texture_heap;
        gpu::uint64 texture_heap_offset;
        gpu::TextureDescriptorHeap* texture_descriptor_heap;
        gpu::uint32 texture_slot;
    };
    BackendUserData user_data {.device = device, .texture_heap = texture_heap, .texture_heap_offset = texture_heap_offset, .texture_descriptor_heap = texture_descriptor_heap, .texture_slot=static_cast<gpu::uint32>(2*cascades.size()+1) };
    ImGui_ImplNoGraphicsAPI_Init({
        .device = device,
        .nearest_sampler_slot = 0,
        .linear_sampler_slot = 1,
        .user_data = &user_data,
        .allocate_frame_memory = [](void* user_data, gpu::uint64 size, gpu::uint64 alignment) {
            BackendUserData* backend_data = static_cast<BackendUserData*>(user_data);
            return allocate<gpu::byte>(backend_data->frame_heap, size, alignment);
        },
        .create_texture = [](void* user_data, const gpu::TextureDesc& desc) {
            BackendUserData* backend_data = static_cast<BackendUserData*>(user_data);
            const gpu::SizeAlign texture_size = gpu::get_texture_size_align(backend_data->device, desc);
            backend_data->texture_heap_offset = align_up(backend_data->texture_heap_offset, texture_size.align);
            gpu::Texture* texture = gpu::create_texture(backend_data->device, desc, backend_data->texture_heap, backend_data->texture_heap_offset);
            backend_data->texture_heap_offset = (backend_data->texture_heap_offset + texture_size.align - 1) & ~(texture_size.align - 1);
            backend_data->texture_heap_offset += texture_size.size;
            const gpu::uint32 texture_slot = backend_data->texture_slot++;
            gpu::write_texture_descriptor(backend_data->texture_descriptor_heap, texture_slot, texture, gpu::TextureDescriptorType::sampled);
            gpu::initialize_texture(backend_data->commands, texture);  // can be done in update_texture
            return ImGui_ImplNoGraphicsAPI_Texture{ .slot = texture_slot, .texture_data = texture };
        },
        .update_texture = [](void* user_data, const ImGui_ImplNoGraphicsAPI_Texture& texture, const gpu::TextureCopyDesc& copy_desc, gpu::Span<gpu::byte> data) {
            BackendUserData* backend_data = static_cast<BackendUserData*>(user_data);
            ImGui_ImplNoGraphicsAPI_UpdateTexture(backend_data->commands, static_cast<gpu::Texture*>(texture.texture_data), copy_desc, data);
        },
        .destroy_texture = [](void* /*user_data*/, const ImGui_ImplNoGraphicsAPI_Texture& texture) {
            gpu::destroy_texture(static_cast<gpu::Texture*>(texture.texture_data));
        }
    });

    ImVec4 clear_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

    while (RGFW_window_shouldClose(window) == RGFW_FALSE) {
        update_surface(window);
        RGFW_pollEvents();

        if (latest_completion.value >= 2) {
            gpu::wait_timeline({ .semaphore = latest_completion.semaphore, .value = latest_completion.value - 1 });
        }
        gpu::uint64 frame_index = latest_completion.value % 2;
        gpu::GpuCpuRange<gpu::byte> frame_heap = {
            .cpu = heap.range.cpu+frame_index*heap.range.size/2,
            .gpu = heap.range.gpu+frame_index*heap.range.size/2,
            .size = heap.range.size/2,
        };
        gpu::CommandPool* command_pool = command_pools[frame_index];
        gpu::reset_command_pool(command_pool);
        gpu::CommandBuffer* commands = gpu::begin_commands(command_pool);
        const gpu::SwapchainFrame frame = gpu::acquire(commands);
        if (!frame.render_view) {
            gpu::end_commands(commands);
            continue;
        }
        
        ImGui_ImplRgfw_NewFrame();
        ImGui_ImplNoGraphicsAPI_NewFrame();
        ImGui::NewFrame();
        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                gpu::uint32 color = static_cast<gpu::uint32>(clear_color.x*255+0.5) << 16 |
                static_cast<gpu::uint32>(clear_color.y*255+0.5) << 8 |
                static_cast<gpu::uint32>(clear_color.z*255+0.5) |
                static_cast<gpu::uint32>(clear_color.w*255+0.5) << 24;
                draw_dot({static_cast<gpu::uint32>(io.MousePos.x), static_cast<gpu::uint32>(io.MousePos.y)}, color, geometry, commands, frame_heap);
            }

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                draw_dot({static_cast<gpu::uint32>(io.MousePos.x), static_cast<gpu::uint32>(io.MousePos.y)}, 0, geometry, commands, frame_heap);
            }
        }
        {
            ImGui::Begin("Hello, world!");
                ImGui::ColorEdit3("clear color", &clear_color.x);
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }
        ImGui::Render();

        gpu::set_texture_descriptor_heap(commands, texture_descriptor_heap);
        gpu::set_sampler_descriptor_heap(commands, sampler_descriptor_heap);

        for (gpu::uint8 direction = 0; direction<4; ++direction) {
            constexpr int n_traced=3;
            for (int i=0; i<n_traced && i<cascades.size(); ++i) {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate<CascadeData>(frame_heap, 1, 16);
                cascade_data.cpu->texture_indices = cascades[i].traces_texture_slot << 8 | (direction << 24);
                bind_pso(commands, compute_trace_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascades[i].width+15)/16, .y = (cascades[i].height+15)/16, .z = cascades[i].cascade_cones+1});
                barrier(commands,
                    gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                    gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            }
            for (int i=n_traced; i<cascades.size(); ++i) {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate<CascadeData>(frame_heap, 1, 16);
                cascade_data.cpu->texture_indices = cascades[i].traces_texture_slot;
                bind_pso(commands, approximate_traces_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascades[i].width+15)/16, .y = (cascades[i].height+15)/16, .z = cascades[i].cascade_cones+1});
                barrier(commands,
                    gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                    gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            }
            {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate<CascadeData>(frame_heap, 1, 16);
                cascade_data.cpu->texture_indices = cascades.back().cones_texture_slot | (cascades.back().traces_texture_slot << 8);
                bind_pso(commands, compute_cone_fluence_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascades.back().width+15)/16, .y = (cascades.back().height+15)/16, .z = cascades.back().cascade_cones});
                barrier(commands,
                    gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                    gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            }
            for (auto i=cascades.size()-1; i-->0;) {
                gpu::GpuCpuRange<CascadeData> cascade_data = allocate<CascadeData>(frame_heap, 1, 16);
                cascade_data.cpu->texture_indices = cascades[i].cones_texture_slot | (cascades[i].traces_texture_slot << 8);
                if (i==0) {
                    cascade_data.cpu->texture_indices |= ((direction+1) << 24);
                }
                bind_pso(commands, accumulate_cone_fluence_pso);
                dispatch(commands, cascade_data.gpu, {.x = (cascades[i].width+15)/16, .y = (cascades[i].height+15)/16, .z = cascades[i].cascade_cones});
                barrier(commands,
                    gpu::Stage::compute, gpu::Access::shader_write | gpu::Access::shader_read,
                    gpu::Stage::compute,  gpu::Access::shader_read | gpu::Access::shader_write);
            }
        }

        {
            gpu::GpuCpuRange<CascadeData> cascade_data = allocate<CascadeData>(frame_heap, 1, 16);

            user_data.commands = commands;
            user_data.frame_heap = frame_heap;
            ImGui_ImplNoGraphicsAPI_UpdateTextures(ImGui::GetDrawData());

            gpu::begin_render_pass(commands, {
                .colors = { {.render_view = frame.render_view, .load = gpu::LoadOp::clear } },
            });

            cascade_data.cpu->texture_indices = (cascades.front().cones_texture_slot << 8);
            //cascade_data.cpu->texture_indices = cascades[0].traces_texture_slot << 8;
            bind_pso(commands, render_cascades_pso);
            draw(commands, cascade_data.gpu, 3);

            ImGui_ImplNoGraphicsAPI_RenderDrawData(commands, ImGui::GetDrawData());

            gpu::end_render_pass(commands);
        }
        gpu::end_commands(commands);
        latest_completion.value++;
        gpu::submit_and_present(device, { .commands = {commands}, .completion = latest_completion });
    }

    gpu::wait_idle(device);

    ImGui_ImplNoGraphicsAPI_Shutdown();
    ImGui_ImplRgfw_Shutdown();
    ImGui::DestroyContext();

    gpu::destroy_pso(compute_trace_pso);
    gpu::destroy_pso(approximate_traces_pso);
    gpu::destroy_pso(compute_cone_fluence_pso);
    gpu::destroy_pso(accumulate_cone_fluence_pso);
    gpu::destroy_pso(render_cascades_pso);
    gpu::destroy_gpu_heap(heap);
    gpu::destroy_texture(geometry);
    for (uint32 i=0; i<cascades.size(); ++i) {
        gpu::destroy_texture(cascades[i].traces_texture);
        gpu::destroy_texture(cascades[i].cones_texture);
    }
    gpu::destroy_texture_heap(texture_heap);
    gpu::destroy_texture_descriptor_heap(texture_descriptor_heap);
    gpu::destroy_sampler_descriptor_heap(sampler_descriptor_heap);
    gpu::destroy_command_pool(command_pools[1]);
    gpu::destroy_command_pool(command_pools[0]);
    gpu::destroy_timeline_semaphore(latest_completion.semaphore);
    gpu::destroy_device(device);
    RGFW_window_close(window);
    return 0;
}
