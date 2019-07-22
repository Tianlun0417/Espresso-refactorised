#include "FloatTypeEspresso/RandomInit.h"


void dense_layer_rand_weight(DenseLayer *den_layer) {
    float *arr_weight = (float *) malloc(den_layer->M * den_layer->N * sizeof(float));

    random_init_arr(arr_weight, den_layer->M * den_layer->N);

    FloatTensor dense_weight = tensor_from_ptr(1, den_layer->M, den_layer->N, 1, arr_weight);
    denseLayer_set(&dense_weight, den_layer);
}

void batchnorm_layer_rand_weight(bnormLayer *bnorm_layer, size_t layer_size) {
    float *arr_bnorm_mean = (float *) calloc(layer_size, sizeof(float));
    float *arr_bnorm_istd = (float *) calloc(layer_size, sizeof(float));
    float *arr_bnorm_gamma = (float *) calloc(layer_size, sizeof(float));
    float *arr_bnorm_beta = (float *) calloc(layer_size, sizeof(float));

    random_init_arr(arr_bnorm_mean, layer_size);
    random_init_arr(arr_bnorm_istd, layer_size);
    random_init_arr(arr_bnorm_gamma, layer_size);
    random_init_arr(arr_bnorm_beta, layer_size);

    FloatTensor bnorm_mean = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_mean);
    FloatTensor bnorm_istd = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_istd);
    FloatTensor bnorm_gamma = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_gamma);
    FloatTensor bnorm_beta = tensor_from_ptr(1, layer_size, 1, 1, arr_bnorm_beta);
    bnormLayer_set(&bnorm_mean, &bnorm_istd, &bnorm_gamma, &bnorm_beta, bnorm_layer);
}

void conv_layer_rand_weight(ConvLayer *conv_layer) {
    // L - no input channels
    // D - no output channels
    // M - kernel height
    // N - kernel width

    size_t tensor_size  = conv_layer->D * conv_layer->M * conv_layer->N * conv_layer->L;
    conv_layer->W.data  = malloc(tensor_size * sizeof(float));
    conv_layer->W.D     = conv_layer->D;
    conv_layer->W.M     = conv_layer->M;
    conv_layer->W.N     = conv_layer->N;
    conv_layer->W.L     = conv_layer->L;
    conv_layer->W.MNL   = conv_layer->M * conv_layer->N * conv_layer->L;
    conv_layer->W.bytes = BYTES(float, tensor_size);

    random_init_arr(conv_layer->W.data, tensor_size);
}

void random_init_arr(float *arr, size_t arr_length) {
    //srand(0);
    for (int i = 0; i < arr_length; i++) {
        if ((float) rand() / (float) (RAND_MAX) > THRESHOLD) arr[i] = 1.0f;
        else arr[i] = 0.0f;
    }
}

void random_init_tensor(int D, int M, int N, int L) {

}