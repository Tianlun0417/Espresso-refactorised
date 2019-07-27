#ifndef BPDENSEOUTPUTLAYER_H
#define BPDENSEOUTPUTLAYER_H

#include "BPTensor.h"
#include "cblas.h"

typedef struct {
    int output_dim, input_dim;
    BPTensor W;
    BPTensor in;
    float *output_arr;
} BPDenseOutputLayer;

void bp_dense_output_layer_init(BPDenseOutputLayer *dense_layer_ptr, int M, int N);

void bp_dense_output_layer_free(BPDenseOutputLayer *dl);

void bp_dense_output_layer_forward(BPTensor *input_tensor, BPDenseOutputLayer *dense_layer, int cpy);


#endif //ESPRESSO_REFACTORISED_BPDENSEOUTPUTLAYER_H
