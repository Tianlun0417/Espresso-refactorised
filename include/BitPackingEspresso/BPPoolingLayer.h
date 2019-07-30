#ifndef BPPOOLINGLAYER_H
#define BPPOOLINGLAYER_H

#include "BPTensor.h"
#include "FloatTypeEspresso/Utilities.h"

typedef enum {
    MAXPOOL, AVGPOOL
} BPPoolingStrategy;

typedef struct {
    int M, N, Stride_m, Stride_n, padding;
    BPPoolingStrategy strategy;
    BPTensor out, mask;
} BPPoolLayer;

void bp_pool_layer_init(BPPoolLayer *pool_layer_ptr, int M, int N, int Stride_m, int Stride_n,
                        int padding, BPPoolingStrategy strategy);

void bp_pool_layer_free(BPPoolLayer *pl);

void bp_pool_layer_forward(BPTensor *t, BPPoolLayer *pl);


#endif //ESPRESSO_REFACTORISED_BPPOOLINGLAYER_H
