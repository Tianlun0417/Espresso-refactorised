#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "IntTensor.h"


#define DENSEL_INIT(dl) (                   \
          dl.M=0, dl.N=0,                   \
          dl.W .data=NULL, dl.b .data=NULL, \
          dl.in.data=NULL, dl.out.data=NULL)


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int M, N;
    IntTensor W, b;
    IntTensor in, out;
} DenseLayer;

DenseLayer denseLayer_init(int M, int N);

void denseLayer_print_shape(DenseLayer *dl);

void denseLayer_free(DenseLayer *dl);

void denseLayer_set(IntTensor *W, DenseLayer *dl);

void denseLayer_forward(IntTensor *input_tensor, DenseLayer *dense_layer, int cpy);


#ifdef __cplusplus
}
#endif
#endif //ESPRESSO_REFACTORISED_DENSELAYER_H
