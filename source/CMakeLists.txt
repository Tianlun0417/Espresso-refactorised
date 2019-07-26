# source files
file(GLOB SRC
        FloatTypeEspresso/*.c
        BitPackingEspresso/*.c
        CNN/*.c)

add_library(espresso_refactorised ${SRC} BitPackingEspresso/BPTensor.c ../include/BitPackingEspresso/BPTensor.h BitPackingEspresso/BPDenseLayer.c ../include/BitPackingEspresso/BPDenseLayer.h BitPackingEspresso/BPGEMM.c ../include/BitPackingEspresso/BPGEMM.h BitPackingEspresso/BPPoolingLayer.c ../include/BitPackingEspresso/BPPoolingLayer.h)