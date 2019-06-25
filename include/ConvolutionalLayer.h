#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include "Tensor.h"
#include "Utilities.h"

#define CONVL_INIT(cl) {                                       \
          cl.D=0; cl.M=0; cl.N=0; cl.L=0;                      \
          cl.Sm=0; cl.Sn=0; cl.do_padding=0;                            \
          cl.W.data=NULL;  cl.b.data=NULL; cl.out.data=NULL;   \
          cl.in.data=NULL;   \
}

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int D, M, N, L, Sm, Sn, do_padding;
    floatTensors W, b, out, in;
    /*floatTensors dW, db;*/ // we don't need the gradients for inference
} convLayer;

convLayer convLayer_init(int Sm, int Sn, int do_padding);
void convLayer_print_shape(convLayer *cl);
void convLayer_free(convLayer *cl);
void convLayer_set(floatTensors *W, convLayer *cl);
void convLayer_forward(floatTensors *t, convLayer *cl, int save);

#ifdef __cplusplus
}
#endif

#endif //C_DEMOS_CONVOLUTIONALLAYER_H
