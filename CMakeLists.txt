cmake_minimum_required(VERSION 3.18)
project(drone_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── CPU Feature detection ──────────────────────────────────────────────────
include(CheckCXXSourceRuns)

# Test AVX2 + FMA availability
set(AVX2_TEST_SRC "
#include <immintrin.h>
int main() {
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_fmadd_pd(a, a, a);
    (void)b; return 0;
}")

if(NOT MSVC)
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
    check_cxx_source_runs("${AVX2_TEST_SRC}" HAS_AVX2_FMA)
    unset(CMAKE_REQUIRED_FLAGS)
else()
    # MSVC: /arch:AVX2 implies FMA
    set(HAS_AVX2_FMA TRUE)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(
        /O2 /Oi /GL /fp:fast
        /W3 /wd4996 /EHsc /MP
    )
    # AVX2 + FMA
    add_compile_options(/arch:AVX2)
    add_compile_definitions(
        _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS
        NOMINMAX WIN32_LEAN_AND_MEAN
        DRONE_AVX2=1
    )
    add_link_options(/LTCG)
else()
    add_compile_options(
        -O3 -march=native -ffast-math
        -Wall -Wextra -Wno-unused-parameter
    )
    add_compile_definitions(_USE_MATH_DEFINES)
    if(HAS_AVX2_FMA)
        add_compile_options(-mavx2 -mfma)
        add_compile_definitions(DRONE_AVX2=1)
        message(STATUS "AVX2 + FMA enabled")
    else()
        message(STATUS "AVX2 not available — using scalar fallback")
    endif()
endif()

# ── Threading ────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── pybind11 ─────────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# ── Build the extension module ────────────────────────────────────────────────
pybind11_add_module(drone_core
    core/bindings.cpp
)

target_include_directories(drone_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(drone_core PRIVATE Threads::Threads)

set_target_properties(drone_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    set_target_properties(drone_core PROPERTIES SUFFIX ".pyd")
endif()

install(TARGETS drone_core DESTINATION .)

# ── Print summary ─────────────────────────────────────────────────────────────
message(STATUS "==============================================")
message(STATUS "  drone_core parallel build config")
message(STATUS "  C++ standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  AVX2+FMA        : ${HAS_AVX2_FMA}")
message(STATUS "  Thread model    : ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "==============================================")
