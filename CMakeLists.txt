cmake_minimum_required(VERSION 3.10.2)
project(espresso_refactorised C)

#find_package(GEMMLOWP REQUIRED)

option(USE_GEMMLOWP OFF)
option(USE_CBLAS ON)

find_package(BLAS REQUIRED)

set(CMAKE_C_STANDARD 99)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("/home/linuxbrew/.linuxbrew/include")

link_directories("/home/linuxbrew/.linuxbrew/lib")

add_subdirectory("source")
#add_subdirectory("test")
add_subdirectory("demo")
