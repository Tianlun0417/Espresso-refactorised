#ifndef BPCONVOLUTIONALLAYER_H
#define BPCONVOLUTIONALLAYER_H

#include "BPTensor.h"
#include "BPGEMM.h"

typedef struct {
    int D, M, N, L, Stride_m, Stride_n, padding;
    BPTensor W, b, out, in;
} BPConvLayer;

void bp_conv_layer_init(BPConvLayer *conv_layer_ptr, int L, int D, int M, int N, int Stride_m, int Stride_n, int padding);

void bp_conv_layer_free(BPConvLayer *cl);

void bp_conv_layer_forward(BPTensor *input_tensor, BPConvLayer *cl, int save);


#endif //ESPRESSO_REFACTORISED_BPCONVOLUTIONALLAYER_H
