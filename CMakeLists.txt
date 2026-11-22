cmake_minimum_required (VERSION 3.31)

project ("radiance_cascade")

if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
endif()
add_subdirectory("external/NoGraphicsAPI")

add_library(RGFW INTERFACE)
target_include_directories(RGFW INTERFACE external/RGFW)

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

add_library(imgui_impl_rgfw INTERFACE)
target_include_directories(imgui_impl_rgfw INTERFACE external/imgui_impl_rgfw)

add_library(imgui_impl_nographicsapi)
target_sources(
    imgui_impl_nographicsapi
    PRIVATE
        external/imgui_impl_nographicsapi/imgui_impl_nographicsapi.cpp
    PUBLIC
        FILE_SET HEADERS
		FILES
			"external/imgui_impl_nographicsapi/imgui_impl_nographicsapi.h"
        BASE_DIRS
            external/imgui_impl_nographicsapi
)
target_include_directories(imgui_impl_nographicsapi PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(imgui_impl_nographicsapi PRIVATE imgui NoGraphicsAPI)


add_executable(radiance_cascade "src/main.cpp")
target_include_directories(radiance_cascade PRIVATE "include")
target_link_libraries(radiance_cascade PRIVATE RGFW NoGraphicsAPI::NoGraphicsAPI imgui imgui_impl_rgfw imgui_impl_nographicsapi)
if(APPLE)
    target_link_libraries(radiance_cascade PRIVATE "-framework Cocoa")
endif()


find_program(SLANGC_EXECUTABLE NAMES slangc HINTS "$ENV{VULKAN_SDK}/bin" REQUIRED)
if(APPLE)
    find_program(XCRUN_EXECUTABLE NAMES xcrun REQUIRED)
    set(SHADER_EXTENSION ".metallib")
else()
    set(SHADER_EXTENSION ".spv")
endif()

function (target_add_slang_shader TARGET SHADER_SOURCE)
    cmake_path(GET SHADER_SOURCE FILENAME SHADER_OUTPUT)
    cmake_path(REPLACE_EXTENSION SHADER_OUTPUT LAST_ONLY ${SHADER_EXTENSION})
    set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_OUTPUT})
    if(APPLE)
        cmake_path(REPLACE_EXTENSION SHADER_OUTPUT LAST_ONLY ".metal" OUTPUT_VARIABLE SHADER_METAL)
        cmake_path(REPLACE_EXTENSION SHADER_OUTPUT LAST_ONLY ".air" OUTPUT_VARIABLE SHADER_AIR)
        add_custom_command (
            OUTPUT  ${SHADER_OUTPUT}
            BYPRODUCTS ${SHADER_METAL} ${SHADER_AIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
            COMMAND ${SLANGC_EXECUTABLE} -target metal -DNOGRAPHICSAPI_METAL -fvk-use-c-layout -matrix-layout-row-major -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/external/NoGraphicsAPI/include" -o ${SHADER_METAL} -depfile ${SHADER_OUTPUT}.d ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
            COMMAND sed -i "" "1s|^[^:]*:|${SHADER_OUTPUT}:|" ${SHADER_OUTPUT}.d  # update the target to this commands output; slangc names the .metal file as the depfile target
            COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -std=metal4.0 -target air64-apple-macos${CMAKE_OSX_DEPLOYMENT_TARGET} -c ${SHADER_METAL} -o ${SHADER_AIR}
            COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${SHADER_AIR} -o ${SHADER_OUTPUT}
            DEPENDS ${SHADER_SOURCE}
            DEPFILE "${SHADER_OUTPUT}.d"
            COMMENT "Compiling Slang Shaders"
            VERBATIM)
    else()
        add_custom_command (
            OUTPUT  ${SHADER_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
            COMMAND ${SLANGC_EXECUTABLE} -target spirv -profile spirv_1_5 -emit-spirv-directly -fvk-use-entrypoint-name -fvk-use-c-layout -matrix-layout-row-major -capability spvDescriptorHeapEXT -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/external/NoGraphicsAPI/include" -o ${SHADER_OUTPUT} -depfile ${SHADER_OUTPUT}.d ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
            DEPENDS ${SHADER_SOURCE}
            DEPFILE "${SHADER_OUTPUT}.d"
            COMMENT "Compiling Slang Shaders"
            VERBATIM)
    endif()
    target_sources(${TARGET} PRIVATE ${SHADER_OUTPUT})
    set_source_files_properties(${SHADER_OUTPUT} PROPERTIES HEADER_FILE_ONLY TRUE)
    source_group("Shaders" FILES ${SHADER_OUTPUT})
    target_compile_definitions(${TARGET} PRIVATE SHADER_EXTENSION="${SHADER_EXTENSION}")
endfunction()

target_add_slang_shader(radiance_cascade "src/shader.slang")
