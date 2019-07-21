#include "IntTypeEspresso/InputLayer.h"


void inputLayer_load(IntTensor *t, InputLayer *il) {
    il->out = tensor_copy(t);
}


void inputLayer_free(InputLayer *il) {
    tensor_free(&il->out);
}

/*
 * input layer binarizes the input image data
 * */
void inputLayer_forward(InputLayer *il) {
    if (!il->out.data) {
        fprintf(stderr, "err: in null\n");
        exit(-1);
    }

    EspInt *ptr = il->out.data;
    const int len = tensor_len(&il->out);
    for (int i = 0; i < len; i++)
        ptr[i] = 2 * ptr[i] / 255 - 1;
}
