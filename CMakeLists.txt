cmake_minimum_required(VERSION 3.18)
project(FUDA_HGEMM LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
# Modify based on your GPU:
# - RTX 5090 Laptop: sm_120 (Blackwell)
# - RTX 40 series: sm_89 (Ada Lovelace)
# - RTX 30 series: sm_80 (Ampere)
# - RTX 20 series: sm_75 (Turing)
# - GTX 10 series: sm_70 (Volta)
set(CMAKE_CUDA_ARCHITECTURES 120)  # For RTX 5090 Laptop (Blackwell)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/native
)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Optional: Add separate kernels as subdirectories if needed
# add_subdirectory(cuda/native)

# Create test executables
# Native kernel test
add_executable(test_hgemm_native
    test/test_hgemm.cu
)

target_link_libraries(test_hgemm_native
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
)

# Set output directory
set_target_properties(test_hgemm_native PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Utility target to build all tests
add_custom_target(tests ALL
    DEPENDS test_hgemm_native
)

# Print build information
message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
