#include "FloatTypeEspresso/InputLayer.h"



void input_layer_load(FloatTensor *in, InputLayer *il) {
    int D = in->D, M = in->M, N = in->N, L = in->L;
    il->out = tensor_init(D, M, N, L);
    free(il->out.data);
    il->out.data = in->data;
}

void input_layer_free(InputLayer *il) {
    tensor_free(&il->out);
}

/*
 * input layer binarizes the input image data
 * */
void input_layer_forward(InputLayer *il) {
    if (!il->out.data) {
        fprintf(stderr, "err: in null\n");
        exit(-1);
    }

    float *ptr = il->out.data;
    const int len = tensor_len(&il->out);
//    for (int i = 0; i < len; i++)
//        ptr[i] = ptr[i] / 255;
    for (int i = 0; i < len; i++)
        if (ptr[i] / 255 > 0.5) ptr[i] = 1;
        else ptr[i] = 0;
}