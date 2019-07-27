#include "BitPackingEspresso/BPRandomInit.h"

void bp_random_init_packed_arr(__uint32_t *arr, size_t arr_packed_len) {
    for (int i = 0; i < arr_packed_len; i++){
        __uint32_t tmp = 0;
        for (int j = 0; j < 32; j++)
            if ((float) rand() / (float) (RAND_MAX) > THRESHOLD)
                tmp = tmp | (1<<i);
        arr[i] = tmp;
    }
}

void bp_dense_layer_rand_weight(BPDenseLayer *den_layer) {
    size_t packed_len = (den_layer->M * den_layer->N) / 32;
    den_layer->W.data = malloc(packed_len * sizeof(__uint32_t));

    bp_random_init_packed_arr(den_layer->W.data, packed_len);
}

void bp_bnorm_layer_rand_weight(BPBnormLayer *bnorm_layer) {
    bp_bnormLayer_free(bnorm_layer);
    bnorm_layer->mean.data  = malloc(bnorm_layer->N * sizeof(__uint32_t));
    bnorm_layer->istd.data  = malloc(bnorm_layer->N * sizeof(__uint32_t));
    bnorm_layer->gamma.data = malloc(bnorm_layer->N * sizeof(__uint32_t));
    bnorm_layer->beta.data  = malloc(bnorm_layer->N * sizeof(__uint32_t));

    bp_random_init_packed_arr(bnorm_layer->mean.data, bnorm_layer->N);
    bp_random_init_packed_arr(bnorm_layer->istd.data, bnorm_layer->N);
    bp_random_init_packed_arr(bnorm_layer->gamma.data, bnorm_layer->N);
    bp_random_init_packed_arr(bnorm_layer->beta.data, bnorm_layer->N);

}

void bp_conv_layer_rand_weight(BPConvLayer *conv_layer) {
    // L - no input channels
    // D - no output channels
    // M - kernel height
    // N - kernel width

    size_t packed_len  = (conv_layer->D * conv_layer->M * conv_layer->N * conv_layer->L) / 32;
    conv_layer->W.data  = malloc(packed_len * sizeof(__uint32_t));
    conv_layer->W.D     = conv_layer->D;
    conv_layer->W.M     = conv_layer->M;
    conv_layer->W.N     = conv_layer->N;
    conv_layer->W.L     = conv_layer->L;
    conv_layer->W.MNL   = conv_layer->M * conv_layer->N * conv_layer->L;
    conv_layer->W.bytes = BYTES(__uint32_t , packed_len);

    bp_random_init_packed_arr(conv_layer->W.data, packed_len);

}

void bp_dense_output_layer_rand_weight(BPDenseOutputLayer *den_layer) {

    size_t packed_len = (den_layer->output_dim * den_layer->input_dim) / 32;
    den_layer->W.data = malloc(packed_len * sizeof(__uint32_t));

    bp_random_init_packed_arr(den_layer->W.data, packed_len);
}
