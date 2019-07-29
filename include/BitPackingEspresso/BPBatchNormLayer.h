#ifndef BPBATCHNORMLAYER_H
#define BPBATCHNORMLAYER_H

#include "BPTensor.h"


typedef struct {
    int N, ug;
    BPTensor mean, istd;
    BPTensor gamma, beta;
    BPTensor in;
} BPBnormLayer;

void bp_bnorm_layer_init(BPBnormLayer *bn_layer, size_t size);

void bp_bnorm_layer_free(BPBnormLayer *bnl);

void bp_bnorm_layer_forward(BPTensor *input_tensor, BPBnormLayer *batchnorm_layer, int save);

#endif //ESPRESSO_REFACTORISED_BPBATCHNORMLAYER_H
