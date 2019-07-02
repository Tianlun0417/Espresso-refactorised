#include "RandomInit.h"


void init_dense_layer(denseLayer *den_layer, int M, int N){
    float* arr_weight = (float*) calloc(M * N, sizeof(float));

    random_init_arr(arr_weight, M * N);

    FloatTensor dense_weight = tensor_from_ptr(1, M, N, 1, arr_weight);
    denseLayer_set(&dense_weight, den_layer);
}

void init_batchnorm_layer(bnormLayer *bnorm_layer, size_t layer_size){
    float* arr_bnorm_mean  = (float*) calloc(layer_size, sizeof(float));
    float* arr_bnorm_istd  = (float*) calloc(layer_size, sizeof(float));
    float* arr_bnorm_gamma = (float*) calloc(layer_size, sizeof(float));
    float* arr_bnorm_beta  = (float*) calloc(layer_size, sizeof(float));

    random_init_arr(arr_bnorm_mean, layer_size);
    random_init_arr(arr_bnorm_istd, layer_size);
    random_init_arr(arr_bnorm_gamma, layer_size);
    random_init_arr(arr_bnorm_beta, layer_size);

    FloatTensor bnorm_mean  = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_mean);
    FloatTensor bnorm_istd  = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_istd);
    FloatTensor bnorm_gamma = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_gamma);
    FloatTensor bnorm_beta  = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_beta);
    bnormLayer_set(&bnorm_mean, &bnorm_istd, &bnorm_gamma, &bnorm_beta, bnorm_layer);

    free(arr_bnorm_mean);
    free(arr_bnorm_istd);
    free(arr_bnorm_gamma);
    free(arr_bnorm_beta);
}

void init_conv_layer(convLayer *conv_layer){
    conv_layer->D = 1;
    conv_layer->M = 128;
    conv_layer->N = 3;
    conv_layer->L = 3;
    conv_layer->Stride_m = 3;
    conv_layer->Stride_n = 1;
    conv_layer->do_padding = 1;
    float arr_W[128];

    for (int i=0; i<128; i++){
        int random_int = rand();
        if (random_int>15000) arr_W[i] = 1;
        else arr_W[i] = 0;
    }

    conv_layer->W.data = arr_W;
}

void random_init_arr(float* arr, size_t arr_length){
    //srand(0);
    for(int i=0; i<arr_length; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr[i] = 1.0f;
        else arr[i] = 0.0f;
    }
}