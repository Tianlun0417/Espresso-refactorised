# source files
file(GLOB SRC *.c)
add_library(espresso_refactorised ${SRC} ../CNNs/AlexNet.c)