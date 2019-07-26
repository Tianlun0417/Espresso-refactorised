#ifndef BPDENSELAYER_H
#define BPDENSELAYER_H

#include "BPTensor.h"
#include "BPGEMM.h"

typedef struct {
    int M, N;
    BPTensor W, b;
    BPTensor in, out;
} BPDenseLayer;

void dense_layer_init(BPDenseLayer *dense_layer_ptr, int M, int N);

void denseLayer_free(BPDenseLayer *dl);

void dense_layer_forward(BPTensor *input_tensor, BPDenseLayer *dense_layer, int cpy);

#endif //ESPRESSO_REFACTORISED_BPDENSELAYER_H
