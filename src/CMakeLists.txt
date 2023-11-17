# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)
# expriment: neeraj
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# expriment: neeraj
project(clip_ggml_library VERSION 0.0.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(BUILD_SHARED_LIBS ON)
add_library(clip_ggml SHARED
  "clip_ggml.cpp"
)

set_target_properties(clip_ggml PROPERTIES
  PUBLIC_HEADER clip.cpp/clip.h 
  OUTPUT_NAME "clip_ggml"
)

add_subdirectory(clip.cpp)

target_compile_definitions(clip_ggml PUBLIC DART_SHARED_LIB)
target_link_libraries(clip_ggml PRIVATE clip ggml ${CMAKE_THREAD_LIBS_INIT})

add_executable(ggmlclip "clip_ggml.cpp")
target_link_libraries(ggmlclip PRIVATE clip ggml)
