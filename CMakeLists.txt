cmake_minimum_required (VERSION 3.31)

project ("radiance_cascade")

if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
endif()
add_subdirectory(external/NoGraphicsAPI)

add_library(RGFW external/RGFW/RGFW.h)
set_source_files_properties(external/RGFW/RGFW.h PROPERTIES LANGUAGE C)
target_compile_definitions(RGFW PUBLIC RGFW_EXPORT PRIVATE RGFW_IMPLEMENTATION)
target_include_directories(RGFW PUBLIC external/RGFW)

add_library(imgui)
file(GLOB IMGUI_SOURCES external/imgui/imgui*.cpp)
file(GLOB IMGUI_HEADERS external/imgui/imgui*.h)
target_sources(
    imgui
    PRIVATE
        ${IMGUI_SOURCES}
    PUBLIC
        FILE_SET HEADERS
		FILES
			${IMGUI_HEADERS}
        BASE_DIRS
            external/imgui
)
target_include_directories(imgui PUBLIC external/imgui/ external/imgui/backends)

add_library(imgui_impl_rgfw external/imgui_impl_rgfw/imgui_impl_rgfw.h)
set_source_files_properties(external/imgui_impl_rgfw/imgui_impl_rgfw.h PROPERTIES LANGUAGE CXX)
target_compile_definitions(imgui_impl_rgfw PRIVATE RGFW_IMGUI_IMPLEMENTATION)
target_include_directories(imgui_impl_rgfw PUBLIC external/imgui_impl_rgfw)
target_link_libraries(imgui_impl_rgfw PRIVATE RGFW imgui)

add_library(imgui_impl_nographicsapi)
target_sources(
    imgui_impl_nographicsapi
    PRIVATE
        external/imgui_impl_nographicsapi/imgui_impl_nographicsapi.cpp
    PUBLIC
        FILE_SET HEADERS
		FILES
			external/imgui_impl_nographicsapi/imgui_impl_nographicsapi.h
        BASE_DIRS
            external/imgui_impl_nographicsapi
)
target_include_directories(imgui_impl_nographicsapi PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(imgui_impl_nographicsapi PRIVATE imgui NoGraphicsAPI)


add_executable(radiance_cascade src/main.cpp)
target_include_directories(radiance_cascade PRIVATE "include")
target_link_libraries(radiance_cascade PRIVATE RGFW NoGraphicsAPI::NoGraphicsAPI imgui imgui_impl_rgfw imgui_impl_nographicsapi)
if(APPLE)
    target_link_libraries(radiance_cascade PRIVATE "-framework Cocoa")
endif()


find_program(SLANGC_EXECUTABLE NAMES slangc HINTS "$ENV{VULKAN_SDK}/bin" REQUIRED)
if(APPLE)
    set(SHADER_EXTENSION ".metallib")
    set(SLANG_TARGET_OPTIONS -target metallib -DNOGRAPHICSAPI_METAL -Xmetal... -std=metal4.0 -target air64-apple-macos${CMAKE_OSX_DEPLOYMENT_TARGET} -X.)
else()
    set(SHADER_EXTENSION ".spv")
    set(SLANG_TARGET_OPTIONS -target spirv -profile spirv_1_5 -emit-spirv-directly -fvk-use-entrypoint-name -capability spvDescriptorHeapEXT)
endif()

function (target_add_slang_shader TARGET SHADER_SOURCE)
    cmake_path(GET SHADER_SOURCE FILENAME SHADER_OUTPUT)
    cmake_path(REPLACE_EXTENSION SHADER_OUTPUT LAST_ONLY ${SHADER_EXTENSION})
    set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_OUTPUT})
    set(SLANG_COMMON_OPTIONS -fvk-use-c-layout -matrix-layout-row-major)
    set(SLANG_INCLUDE_OPTIONS "$<LIST:TRANSFORM,$<LIST:FILTER,$<TARGET_PROPERTY:${TARGET},INCLUDE_DIRECTORIES>,INCLUDE,.>,PREPEND,-I>")
    add_custom_command (
        OUTPUT  ${SHADER_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
        COMMAND ${SLANGC_EXECUTABLE} ${SLANG_TARGET_OPTIONS} ${SLANG_COMMON_OPTIONS} ${SLANG_INCLUDE_OPTIONS} -o ${SHADER_OUTPUT} -depfile ${SHADER_OUTPUT}.d ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
        DEPENDS ${SHADER_SOURCE}
        DEPFILE "${SHADER_OUTPUT}.d"
        COMMENT "Compiling Slang Shaders"
        VERBATIM
        COMMAND_EXPAND_LISTS)
    target_sources(${TARGET} PRIVATE ${SHADER_OUTPUT})
    set_source_files_properties(${SHADER_OUTPUT} PROPERTIES HEADER_FILE_ONLY TRUE)
    source_group("Shaders" FILES ${SHADER_OUTPUT})
    target_compile_definitions(${TARGET} PRIVATE SHADER_EXTENSION="${SHADER_EXTENSION}")
endfunction()

target_add_slang_shader(radiance_cascade src/shader.slang)
