# source files
file(GLOB SRC
        CNN/*.c
        FloatTypeEspresso/*.c
        IntTypeEspresso/*.c)

add_library(espresso_refactorised ${SRC})