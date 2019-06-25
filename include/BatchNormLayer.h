#ifndef BATCHNORMLAYER_H
#define BATCHNORMLAYER_H
#include "Tensor.h"

#define BNORML_INIT(bnl) {                                       \
          bnl.N=0; bnl.ug=0;                                    \
          bnl.mean. data=NULL; bnl.istd  .data=NULL;            \
          bnl.gmean.data=NULL; bnl.gistd .data=NULL;            \
          bnl.beta .data=NULL; bnl.gamma .data=NULL;            \
          bnl.dbeta.data=NULL; bnl.dgamma.data=NULL;            \
          bnl.in.   data=NULL; bnl.tmp.   data=NULL;            \
     }


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int N, ug;
    floatTensors mean,  istd, gmean,  gistd;
    floatTensors gamma, beta, dgamma, dbeta;
    floatTensors in, tmp;
} bnormLayer;


bnormLayer bnormLayer_init(int use_global);
void bnormLayer_free(bnormLayer *bnl);
void bnormLayer_forward(floatTensors *input_tensor, bnormLayer *batchnorm_layer, int save);
void bnormLayer_backward(floatTensors *dt, bnormLayer *bnl);
void bnormLayer_update(bnormLayer *bnl);
void bnormLayer_set(floatTensors *mean,  floatTensors *istd,
                    floatTensors *gamma, floatTensors *beta, bnormLayer *bnl);


#ifdef __cplusplus
}
#endif
#endif //BATCHNORMLAYER_H
