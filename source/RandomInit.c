#include "RandomInit.h"

void init_float_tensor(int D, int M, int N, int L){
    exit(-1);
}

void init_dense_layer(denseLayer *den_layer){
    exit(-1);
}

void init_batchnorm_layer(bnormLayer *batchnorm_layer){
    exit(-1);
}

void init_conv_layer(convLayer *conv_layer){
    conv_layer->D = 1;
    conv_layer->M = 128;
    conv_layer->N = 3;
    conv_layer->L = 3;
    conv_layer->Sm = 3;
    conv_layer->Sn = 1;
    conv_layer->do_padding = 1;
    float arr_W[128];

    for (int i=0; i<128; i++){
        int random_int = rand();
        if (random_int>15000) arr_W[i] = 1;
        else arr_W[i] = 0;
    }

    conv_layer->W.data = arr_W;
}