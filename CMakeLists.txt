cmake_minimum_required(VERSION 3.24)
project(h3metal LANGUAGES C CXX OBJCXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  message(FATAL_ERROR "h3metal targets Apple Silicon Macs only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")

# The GPU runtime the Python package drives through ctypes. Metal kernels are
# compiled at run time from kernels/, so building needs no Metal toolchain.
add_library(h3metal SHARED src/metal_rt.mm src/capi.mm)
target_include_directories(h3metal PUBLIC src)
target_link_libraries(h3metal PRIVATE "-framework Metal" "-framework Foundation")

# Hardware probe
add_executable(gemm_probe bench/hw/gemm_probe.mm)
target_link_libraries(gemm_probe PRIVATE "-framework Metal" "-framework Foundation")
