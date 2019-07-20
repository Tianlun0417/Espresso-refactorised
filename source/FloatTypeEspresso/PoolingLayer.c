#include "FloatTypeEspresso/PoolingLayer.h"


poolLayer poolLayer_init(int M, int N, int Sm, int Sn, poolingStrategy strategy) {
    poolLayer pl = {M, N, Sm, Sn, strategy};
    pl.out.data = NULL;
    pl.mask.data = NULL;
    return pl;
}


poolLayer *new_pool_layer(int M, int N, int Stride_m,
                          int Stride_n, int padding, poolingStrategy strategy) {
    poolLayer *pool_layer_ptr = (poolLayer *) malloc(sizeof(poolLayer));
    pool_layer_ptr->M = M;
    pool_layer_ptr->N = N;
    pool_layer_ptr->Stride_m = Stride_m;
    pool_layer_ptr->Stride_n = Stride_n;
    pool_layer_ptr->strategy = strategy;
    pool_layer_ptr->padding  = padding;
    pool_layer_ptr->out.data = NULL;
    pool_layer_ptr->mask.data = NULL;

    return pool_layer_ptr;
}


void poolLayer_free(poolLayer *pl) {
    tensor_free(&pl->out);
    tensor_free(&pl->mask);
}


void poolLayer_forward(FloatTensor *t, poolLayer *pl) {
    const int W = pl->M, H = pl->N, Sy = pl->Stride_m, Sx = pl->Stride_n;
    const int D = t->D, L = t->L, Ms = t->M, Ns = t->N;
    const int Md = OUT_LEN(Ms, W, Sy);
    const int Nd = OUT_LEN(Ns, H, Sx);

    if (!pl->out.data) pl->out = tensor_init(D, Md, Nd, L);

    if (pl->strategy == MAXPOOL)
        tensor_maxpool(t, &pl->out, W, H, Sx, Sy);
    else
        tensor_avgpool(t, &pl->out, W, H, Sx, Sy);
}


void poolLayer_backward(FloatTensor *dout, poolLayer *pl) {
    exit(-2);
}

void set_pooling_strategy(poolLayer *pl, int strategy) {
    pl->strategy = strategy;
}