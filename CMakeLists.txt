cmake_minimum_required (VERSION 3.31)

project ("radiance_cascade")

add_subdirectory("external/NoGraphicsAPI")

add_library(RGFW INTERFACE)
target_include_directories(RGFW INTERFACE external/RGFW)

add_executable(radiance_cascade "src/main.cpp")
target_include_directories(radiance_cascade PRIVATE "include")
target_link_libraries(radiance_cascade PRIVATE RGFW NoGraphicsAPI::NoGraphicsAPI)
if(APPLE)
    target_link_libraries(radiance_cascade PRIVATE "-framework Cocoa")
endif()

find_program(SLANGC_EXECUTABLE NAMES slangc HINTS "$ENV{VULKAN_SDK}/bin" REQUIRED)
function (target_add_slang_shader TARGET SHADER_SOURCE)
    cmake_path(GET SHADER_SOURCE FILENAME SHADER_OUTPUT)
    cmake_path(REPLACE_EXTENSION SHADER_OUTPUT LAST_ONLY ".spv")
    set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_OUTPUT})
    add_custom_command (
        OUTPUT  ${SHADER_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
        COMMAND ${SLANGC_EXECUTABLE} -target spirv -profile spirv_1_5 -emit-spirv-directly -fvk-use-entrypoint-name -fvk-use-c-layout -matrix-layout-row-major -capability spvDescriptorHeapEXT -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/external/NoGraphicsAPI/include" -o ${SHADER_OUTPUT} -depfile ${SHADER_OUTPUT}.d ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
        DEPENDS ${SHADER_SOURCE}
        DEPFILE "${SHADER_OUTPUT}.d"
        COMMENT "Compiling Slang Shaders"
        VERBATIM)
    target_sources(${TARGET} PRIVATE ${SHADER_OUTPUT})
    set_source_files_properties(${SHADER_OUTPUT} PROPERTIES HEADER_FILE_ONLY TRUE)
    source_group("Shaders" FILES ${SHADER_OUTPUT})
endfunction()

target_add_slang_shader(radiance_cascade "src/shader.slang")
