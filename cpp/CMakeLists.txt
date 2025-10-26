cmake_minimum_required(VERSION 3.10)
project(dataset_reader)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use included Eigen (no need to find_package)
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/eigen)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Create library
add_library(dataset_reader
    src/dataset_reader.cpp
)




# Create comprehensive test executable
add_executable(test_all_datasets
    test/test_all_datasets.cpp
)

target_link_libraries(test_all_datasets
    dataset_reader
)

# Set output directory
set_target_properties(test_all_datasets PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)