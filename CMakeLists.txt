cmake_minimum_required(VERSION 3.10.2)
project(espresso_refactorised C)

find_package(BLAS REQUIRED)

set(CMAKE_C_STANDARD 99)

include_directories("${CMAKE_SOURCE_DIR}/include")

add_subdirectory("source")
# add_subdirectory("test")
add_subdirectory("CNNs")
