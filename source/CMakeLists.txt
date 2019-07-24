# source files
file(GLOB SRC
        FloatTypeEspresso/*.c
        IntTypeEspresso/*.c
        CNN/*.c)

add_library(espresso_refactorised ${SRC})