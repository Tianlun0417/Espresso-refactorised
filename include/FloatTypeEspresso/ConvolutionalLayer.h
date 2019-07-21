#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include "FloatTensor.h"
#include "Utilities.h"

#define CONVL_INIT(cl) {                                       \
          cl.D=0; cl.M=0; cl.N=0; cl.L=0;                      \
          cl.Stride_m=0; cl.Stride_n=0; cl.padding=0;                            \
          cl.W.data=NULL;  cl.b.data=NULL; cl.out.data=NULL;   \
          cl.in.data=NULL;   \
}

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int D, M, N, L, Stride_m, Stride_n, padding;
    FloatTensor W, b, out, in;
    /*FloatTensor dW, db;*/ // we don't need the gradients for inference
} ConvLayer;

ConvLayer convLayer_init(int Sm, int Sn, int padding);

ConvLayer * new_conv_layer(int L, int D, int M, int N, int Stride_m, int Stride_n, int padding);

void convLayer_print_shape(ConvLayer *cl);

void convLayer_free(ConvLayer *cl);

void convLayer_set(FloatTensor *W, ConvLayer *cl);

void conv_layer_forward(FloatTensor *input_t, ConvLayer *cl, int save);

#ifdef __cplusplus
}
#endif

#endif //C_DEMOS_CONVOLUTIONALLAYER_H
