#ifndef BATCHNORMLAYER_H
#define BATCHNORMLAYER_H

#include "FloatTensor.h"

#define BNORML_INIT(bnl) {                                      \
          bnl.N=0; bnl.ug=0;                                    \
          bnl.mean. data=NULL; bnl.istd  .data=NULL;            \
          bnl.beta .data=NULL; bnl.gamma .data=NULL;            \
          bnl.in.   data=NULL;                                  \
     }


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int N, ug;
    FloatTensor mean, istd;
    FloatTensor gamma, beta;
    FloatTensor in;
} BnormLayer;


//BnormLayer bnormLayer_init(int use_global);

void bnorm_layer_init(BnormLayer *bn_layer_ptr, size_t size);

void bnormLayer_free(BnormLayer *bnl);

void bnormLayer_forward(FloatTensor *input_tensor, BnormLayer *batchnorm_layer, int save);

void bnormLayer_backward(FloatTensor *dt, BnormLayer *bnl);

void bnormLayer_update(BnormLayer *bnl);

void bnormLayer_set(FloatTensor *mean, FloatTensor *istd,
                    FloatTensor *gamma, FloatTensor *beta, BnormLayer *bnl);


#ifdef __cplusplus
}
#endif
#endif //BATCHNORMLAYER_H
