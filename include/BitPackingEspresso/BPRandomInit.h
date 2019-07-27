#ifndef BPRANDOMINIT_H
#define BPRANDOMINIT_H

#include "BPTensor.h"
#include "BPBatchNormLayer.h"
#include "BPDenseLayer.h"
#include "BPDenseOutputLayer.h"
#include "BPConvolutionalLayer.h"
#define THRESHOLD 0.6

void bp_dense_layer_rand_weight(BPDenseLayer *den_layer);

void bp_dense_output_layer_rand_weight(BPDenseOutputLayer *den_layer);

void bp_bnorm_layer_rand_weight(BPBnormLayer *bnorm_layer);

void bp_conv_layer_rand_weight(BPConvLayer *conv_layer);

void bp_random_init_packed_arr(__uint32_t *arr, size_t arr_packed_len);


#endif //ESPRESSO_REFACTORISED_BPRANDOMINIT_H
