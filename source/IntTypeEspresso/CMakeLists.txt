# source files
file(GLOB SRC *.c)
add_library(espresso_refactorised ${SRC} ../../include/IntTypeEspresso/Utilities.h)