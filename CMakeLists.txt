cmake_minimum_required(VERSION 3.18)

project(Portfolio_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_PREFIX_PATH "D:/libtorch-2.8.0")

find_package(Torch REQUIRED)

add_executable(ai_inference main.cpp)

target_link_libraries(ai_inference PRIVATE "${TORCH_LIBRARIES}")

if(MSVC)

    target_compile_options(ai_inference PRIVATE /wd4267)

    
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    
    add_custom_command(
        TARGET ai_inference
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:ai_inference>
    )
endif()