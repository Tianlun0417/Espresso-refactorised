#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "FloatTensor.h"


#define DENSEL_INIT(dl) (                   \
          dl.M=0, dl.N=0,                   \
          dl.W .data=NULL, dl.b .data=NULL, \
          dl.in.data=NULL, dl.out.data=NULL)


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int M, N;
    FloatTensor W, b;
    FloatTensor in, out;
} denseLayer;

denseLayer denseLayer_init(int M, int N);

denseLayer * new_dense_layer(int M, int N);

void denseLayer_print_shape(denseLayer *dl);

void denseLayer_free(denseLayer *dl);

void denseLayer_set(FloatTensor *W, denseLayer *dl);

void denseLayer_forward(FloatTensor *input_tensor, denseLayer *dense_layer, int cpy);

void denseLayer_backward(FloatTensor *dt, denseLayer *dl);


#ifdef __cplusplus
}
#endif
#endif //DENSELAYER_H
