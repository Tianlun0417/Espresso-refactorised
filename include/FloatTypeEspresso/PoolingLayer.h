#ifndef ESPRESSO_REFACTORISED_POOLINGLAYER_H
#define ESPRESSO_REFACTORISED_POOLINGLAYER_H

#include "FloatTensor.h"
#include "Utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MAXPOOL, AVGPOOL
} poolingStrategy;

typedef struct {
    int M, N, Stride_m, Stride_n, padding;
    poolingStrategy strategy;
    FloatTensor out, mask;
} PoolLayer;


PoolLayer poolLayer_init(int M, int N, int Sm, int Sn, poolingStrategy strategy);

PoolLayer *new_pool_layer(int M, int N, int Stride_m, int Stride_n, int padding, poolingStrategy strategy);

void poolLayer_free(PoolLayer *pl);

void pool_layer_forward(FloatTensor *t, PoolLayer *pl);

void poolLayer_backward(FloatTensor *dout, PoolLayer *pl);

void set_pooling_strategy(PoolLayer *pl, int strategy);


#ifdef __cplusplus
}
#endif

#endif //ESPRESSO_REFACTORISED_POOLINGLAYER_H
