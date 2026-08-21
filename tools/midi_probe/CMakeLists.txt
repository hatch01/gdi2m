cmake_minimum_required(VERSION 3.20)
project(midi_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(midi_probe main.cpp)
target_link_libraries(midi_probe PRIVATE winmm)

if(MSVC)
    target_compile_options(midi_probe PRIVATE /W4 /permissive-)
endif()

set_target_properties(midi_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR})
