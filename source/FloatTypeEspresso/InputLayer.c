#include "FloatTypeEspresso/InputLayer.h"
#include "FloatTypeEspresso/Utilities.h"


void inputLayer_load(FloatTensor *t, inputLayer *il) {
    il->out = tensor_copy(t);
}


void inputLayer_free(inputLayer *il) {
    tensor_free(&il->out);
}

/*
 * input layer binarizes the input image data
 * */
void inputLayer_forward(inputLayer *il) {
    if (!il->out.data) {
        fprintf(stderr, "err: in null\n");
        exit(-1);
    }

    float *ptr = il->out.data;
    const int len = tensor_len(&il->out);
    for (int i = 0; i < len; i++)
        ptr[i] = 2.0f * ptr[i] / 255 - 1.0f;
}