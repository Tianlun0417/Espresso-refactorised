# source files
file(GLOB SRC
        FloatTypeEspresso/*.c
        IntTypeEspresso/*.c
        CNN/*.c)

add_library(espresso_refactorised ${SRC} CNN/DenseNet.c ../include/CNN/DenseNet.h CNN/AlexNet.c ../include/CNN/AlexNet.h CNN/VGG.c ../include/CNN/VGG.h)