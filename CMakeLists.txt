cmake_minimum_required(VERSION 3.10.2)
project(espresso_refactorised C)

#find_package(GEMMLOWP REQUIRED)

option(USE_GEMMLOWP OFF)
option(USE_CBLAS ON)


if (USE_CBLAS)
    find_package(BLAS REQUIRED)
endif ()

set(CMAKE_C_STANDARD 99)

include_directories("${CMAKE_SOURCE_DIR}/include")

add_subdirectory("source")
add_subdirectory("test")
#add_subdirectory("CNNs")
