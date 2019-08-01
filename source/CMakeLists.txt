# source files
file(GLOB SRC
        FloatTypeEspresso/*.c
        BitPackingEspresso/*.c
        CNN/*.c
        DataLoader/*.c)

add_library(espresso_refactorised ${SRC} CNN/BPResNet.c ../include/CNN/BPResNet.h CNN/BPSqueezeNet.c ../include/CNN/BPSqueezeNet.h CNN/BPVGG.c ../include/CNN/BPVGG.h)
