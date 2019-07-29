# source files
file(GLOB SRC
        FloatTypeEspresso/*.c
        BitPackingEspresso/*.c
        CNN/*.c
        DataLoader/*.c)

add_library(espresso_refactorised ${SRC})
