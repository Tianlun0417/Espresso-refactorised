#include "FloatTypeEspresso/RandomInit.h"


void dense_layer_rand_weight(DenseLayer *den_layer) {
    den_layer->W.data = (float *) malloc(den_layer->M * den_layer->N * sizeof(float));

    random_init_arr(den_layer->W.data, den_layer->M * den_layer->N);
}

void bnorm_layer_rand_weight(bnormLayer *bnorm_layer) {
    bnormLayer_free(bnorm_layer);
    bnorm_layer->mean.data  = malloc(bnorm_layer->N * sizeof(float));
    bnorm_layer->istd.data  = malloc(bnorm_layer->N * sizeof(float));
    bnorm_layer->gamma.data = malloc(bnorm_layer->N * sizeof(float));
    bnorm_layer->beta.data  = malloc(bnorm_layer->N * sizeof(float));
    random_init_arr(bnorm_layer->mean.data, bnorm_layer->N);
    random_init_arr(bnorm_layer->istd.data, bnorm_layer->N);
    random_init_arr(bnorm_layer->gamma.data, bnorm_layer->N);
    random_init_arr(bnorm_layer->beta.data, bnorm_layer->N);
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
        else if((float) rand() / (float) (RAND_MAX) < THRESHOLD &&
        (float) rand() / (float) (RAND_MAX) > 0.3)
            arr[i] = -1.0f;
        else arr[i] = 0.0f;
    }
}

void random_init_tensor(int D, int M, int N, int L) {

}