#include "RGFW.h"
#include "imgui_impl_rgfw.h"
#include "imgui_impl_nographicsapi.h"

#include "NoGraphicsAPI/NoGraphicsAPI.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

#include "shared_types.h"
#include "cascades.h"


#define DOT_SIZE 10

namespace {
#if defined(__APPLE__)
    extern "C" void* sel_getUid(const char*);
    extern "C" void* objc_msgSend();
    extern "C" void* sel_registerName(const char*);

    void* get_metal_layer(RGFW_window* window) {
        void* view = RGFW_window_getView_OSX(window);
        return reinterpret_cast<void*(*)(void*, void*)>(objc_msgSend)(view, sel_getUid("layer"));
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
        void* layer = get_metal_layer(window);
        reinterpret_cast<void (*)(void*, void*, double)>(objc_msgSend)(layer, sel_registerName("setContentsScale:"), RGFW_window_getMonitor(window)->pixelRatio);

        int pixel_width, pixel_height;
        RGFW_window_getSizeInPixels(window, &pixel_width, &pixel_height);
        reinterpret_cast<void (*)(void*, void*, CGSize)>(objc_msgSend)(layer, sel_registerName("setDrawableSize:"), CGSize{static_cast<double>(pixel_width), static_cast<double>(pixel_height)});
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
                4*(i*i+j*j) < dot_size*dot_size ? color : 0u;
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
            .offset = {.x = pos.x-dot_size/2, .y = pos.y-dot_size/2},
            .extent = {.x = dot_size, .y = dot_size, .z = 1}
        });
    }
}

int main() {
    constexpr gpu::uint32 window_width = 1240, window_height = 960;
    constexpr gpu::uint32 world_width = window_width, world_height = window_height;
    constexpr gpu::uint32 cascade0_width = world_width, cascade0_height = world_height;

    RGFW_init("radiance-cascade", 0);
    RGFW_window* window = RGFW_createWindow("radiance-cascade", 0, 0, window_width, window_height, RGFW_windowCenter);

    gpu::Device* device = gpu::create_device({ .window = init_surface(window), .swapchain_format = gpu::Format::bgra8_srgb }).device;

    gpu::TimelinePoint latest_completion{ .semaphore = gpu::create_timeline_semaphore(device) };
    gpu::CommandPool* command_pools[] = { gpu::create_command_pool(device), gpu::create_command_pool(device) };
    gpu::GpuHeap heap = gpu::create_gpu_heap(device, 32*1024*1024);

    gpu::TextureHeap texture_heap = gpu::create_texture_heap(device, 1024*1024*1024);
    gpu::TextureDescriptorHeap* texture_descriptor_heap = gpu::create_texture_descriptor_heap(device, 32);
    gpu::SamplerDescriptorHeap* sampler_descriptor_heap = gpu::create_sampler_descriptor_heap(device, 2);

    struct BackendUserData {
        gpu::Device* device;
        gpu::CommandBuffer* commands;
        gpu::GpuCpuRange<byte> frame_heap;
        gpu::TextureHeap texture_heap;
        gpu::uint64 texture_heap_offset;
        gpu::TextureDescriptorHeap* texture_descriptor_heap;
        gpu::uint16 texture_slot;
        std::vector<gpu::byte> shader;

        gpu::Texture* allocate_texture(const gpu::TextureDesc& texture_desc) {
            gpu::SizeAlign texture_requirements = gpu::get_texture_size_align(device, texture_desc);
            texture_heap_offset = align_up(texture_heap_offset, texture_requirements.align);
            gpu::Texture* texture = gpu::create_texture(device, texture_desc, texture_heap, texture_heap_offset);
            texture_heap_offset += texture_requirements.size;
            return texture;
        }

        gpu::uint16 write_texture_descriptor(gpu::Texture* texture, gpu::TextureDescriptorType type) {
            gpu::write_texture_descriptor(texture_descriptor_heap, texture_slot, texture, type);
            return texture_slot++;
        }
    };

    BackendUserData user_data {
        .device = device,
        .texture_heap = texture_heap,
        .texture_heap_offset = 0,
        .texture_descriptor_heap = texture_descriptor_heap,
        .texture_slot = 0,
        .shader = read_file("shaders/shader" SHADER_EXTENSION)
    };
    
    gpu::TextureDesc texture_desc{
        .extent={.x = world_width, .y = world_height, .z = 1},
        .format = gpu::Format::bgra8_unorm,
        .usage = gpu::TextureUsage::sampled | gpu::TextureUsage::transfer_destination
    };

    gpu::Texture* world_texture = user_data.allocate_texture(texture_desc);
    gpu::uint16 world_texture_id = user_data.write_texture_descriptor(world_texture, gpu::TextureDescriptorType::sampled);

    cascades::Cascades* cascades;

    {
        gpu::CommandBuffer* texture_commands = gpu::begin_commands(command_pools[0]);
        
        gpu::initialize_texture(texture_commands, world_texture);
        gpu::GpuCpuRange<gpu::byte> init_heap = heap.range;
        gpu::GpuCpuRange<gpu::uint32> zeros = allocate<gpu::uint32>(init_heap, world_width*world_height);
        std::memset(zeros.cpu, 0, zeros.size);
        gpu::copy_memory_to_texture(texture_commands, gpu::gpu_range(zeros), world_texture);

        user_data.commands = texture_commands;
        user_data.frame_heap = init_heap;
        cascades = cascades::create_cascades({
            .user_data = &user_data,
            .device = device,
            .cascade0_size = {cascade0_width, cascade0_height},
            .load_shader = [](void* user_data, const char*) {
                BackendUserData* backend_data = static_cast<BackendUserData*>(user_data);
                return gpu::ByteSpan(backend_data->shader.data(), backend_data->shader.size());
            },
            .allocate_texture = [](void* user_data, const gpu::TextureDesc& desc) {
                BackendUserData* backend_data = static_cast<BackendUserData*>(user_data);
                auto texture = backend_data->allocate_texture(desc);
                gpu::initialize_texture(backend_data->commands, texture);
                return backend_data->write_texture_descriptor(texture, gpu::TextureDescriptorType::storage);
            },
            .allocate_frame_data = [](void* user_data, gpu::uint64 size, gpu::uint64 alignment) {
                BackendUserData* backend_data = static_cast<BackendUserData*>(user_data);
                return allocate<gpu::byte>(backend_data->frame_heap, size, alignment);
            },
            .free_texture = [](void*, gpu::uint16) {},
        });

        gpu::end_commands(texture_commands);
        latest_completion.value++;
        gpu::submit(device, {.commands = {texture_commands}, .completion = latest_completion});
    }

    gpu::PSO* render_cascades_pso = create_graphics_pso(device, {
        .vertex = {.code = {user_data.shader.data(), user_data.shader.size()}, .entry_point = "main_vertex"},
        .fragment = {.code = {user_data.shader.data(), user_data.shader.size()}, .entry_point = "main_fragment"},
        .color_targets = { { .format = gpu::Format::bgra8_srgb } }
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
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_IsSRGB;

    ImGui::StyleColorsDark();
    // convert_imgui_style_to_linear_HACK();  // TODO: do this in the shader?

    ImGui_ImplRgfw_InitForOther(window, true);
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
            auto texture = backend_data->allocate_texture(desc);
            const gpu::uint16 texture_slot = backend_data->write_texture_descriptor(texture, gpu::TextureDescriptorType::sampled);
            gpu::initialize_texture(backend_data->commands, texture);
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
        gpu::CommandPool* command_pool = command_pools[frame_index];
        gpu::reset_command_pool(command_pool);
        gpu::CommandBuffer* commands = gpu::begin_commands(command_pool);
        const gpu::SwapchainFrame frame = gpu::acquire(commands);
        if (!frame.render_view) {
            gpu::end_commands(commands);
            continue;
        }

        user_data.commands = commands;
        user_data.frame_heap = {
            .cpu = heap.range.cpu+frame_index*heap.range.size/2,
            .gpu = heap.range.gpu+frame_index*heap.range.size/2,
            .size = heap.range.size/2,
        };
        
        ImGui_ImplRgfw_NewFrame();
        ImGui_ImplNoGraphicsAPI_NewFrame();
        ImGui::NewFrame();
        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                gpu::uint32 color = static_cast<gpu::uint32>(clear_color.x*255+0.5) << 16 |
                static_cast<gpu::uint32>(clear_color.y*255+0.5) << 8 |
                static_cast<gpu::uint32>(clear_color.z*255+0.5) |
                static_cast<gpu::uint32>(clear_color.w*255+0.5) << 24;
                draw_dot({static_cast<gpu::uint32>(io.MousePos.x*world_width/window_width), static_cast<gpu::uint32>(io.MousePos.y*world_height/window_height)}, color, world_texture, commands, user_data.frame_heap);
            }

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                draw_dot({static_cast<gpu::uint32>(io.MousePos.x*world_width/window_width), static_cast<gpu::uint32>(io.MousePos.y*world_height/window_height)}, 0, world_texture, commands, user_data.frame_heap);
            }
        }
        {
            ImGui::Begin("Settings");
                ImGui::ColorEdit3("clear color", &clear_color.x);
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }
        ImGui::Render();

        gpu::set_texture_descriptor_heap(commands, texture_descriptor_heap);
        gpu::set_sampler_descriptor_heap(commands, sampler_descriptor_heap);

        barrier(commands,
            gpu::Stage::transfer | gpu::Stage::compute | gpu::Stage::fragment,
            gpu::Access::transfer_write | gpu::Access::shader_read | gpu::Access::shader_write,
            gpu::Stage::compute, gpu::Access::shader_read | gpu::Access::shader_write);

        gpu::uint16 cascade0_id = cascades::compute_radiance(cascades, world_texture_id, {world_width, world_height}, commands);

        barrier(commands,
            gpu::Stage::compute, gpu::Access::shader_write,
            gpu::Stage::fragment, gpu::Access::shader_read);
        {
            gpu::GpuCpuRange<CascadeData> cascade_data = allocate<CascadeData>(user_data.frame_heap, 1, 16);

            ImGui_ImplNoGraphicsAPI_UpdateTextures(ImGui::GetDrawData());

            gpu::begin_render_pass(commands, {
                .colors = { {.render_view = frame.render_view, .load = gpu::LoadOp::clear } },
            });

            cascade_data.cpu->texture_indices = world_texture_id | (cascade0_id << 8);
            cascade_data.cpu->width1 = cascade0_width;
            cascade_data.cpu->height1 = cascade0_height;
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

    cascades::destroy_cascades(cascades);

    gpu::destroy_pso(render_cascades_pso);
    gpu::destroy_gpu_heap(heap);
    gpu::destroy_texture(world_texture);

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
