# source files
file(GLOB SRC *.c)
add_library(espresso_refactorised ${SRC} ../IntTypeEspresso/IntTensor.c ../../include/IntTypeEspresso/IntTensor.h)