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
} DenseLayer;

//DenseLayer denseLayer_init(int M, int N);

void dense_layer_init(DenseLayer *dense_layer_ptr, int M, int N);

void denseLayer_print_shape(DenseLayer *dl);

void dense_layer_free(DenseLayer *dl);

void denseLayer_set(FloatTensor *W, DenseLayer *dl);

void dense_layer_forward(FloatTensor *input_tensor, DenseLayer *dense_layer, int cpy);

void denseLayer_backward(FloatTensor *dt, DenseLayer *dl);


#ifdef __cplusplus
}
#endif
#endif //DENSELAYER_H
