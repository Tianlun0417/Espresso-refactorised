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
} denseLayer;

denseLayer denseLayer_init(int M, int N);
void denseLayer_print_shape(denseLayer *dl);
void denseLayer_free(denseLayer *dl);
void denseLayer_set(IntTensor *W, denseLayer *dl);
void denseLayer_forward(IntTensor *input_tensor, denseLayer *dense_layer, int cpy);


#ifdef __cplusplus
}
#endif
#endif //ESPRESSO_REFACTORISED_DENSELAYER_H
