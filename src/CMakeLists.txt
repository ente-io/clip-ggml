# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(clip_ggml_library VERSION 0.0.1 LANGUAGES C CXX)

add_library(clip_ggml SHARED
  "clip_ggml.cpp"
)

target_compile_definitions(clip_ggml PUBLIC DART_SHARED_LIB)
# Add target link libraries later
