#include "FloatTypeEspresso/PoolingLayer.h"


PoolLayer poolLayer_init(int M, int N, int Sm, int Sn, PoolingStrategy strategy) {
    PoolLayer pl = {M, N, Sm, Sn, strategy};
    pl.out.data = NULL;
    pl.mask.data = NULL;
    return pl;
}


void pool_layer_init(PoolLayer *pool_layer_ptr, int M, int N, int Stride_m,
                     int Stride_n, int padding, PoolingStrategy strategy) {
    pool_layer_ptr->M = M;
    pool_layer_ptr->N = N;
    pool_layer_ptr->Stride_m = Stride_m;
    pool_layer_ptr->Stride_n = Stride_n;
    pool_layer_ptr->strategy = strategy;
    pool_layer_ptr->padding  = padding;
    pool_layer_ptr->out.data = NULL;
    pool_layer_ptr->mask.data = NULL;
}


void poolLayer_free(PoolLayer *pl) {
    tensor_free(&pl->out);
    tensor_free(&pl->mask);
}


void pool_layer_forward(FloatTensor *t, PoolLayer *pl) {
    const int W = pl->M, H = pl->N, Sy = pl->Stride_m, Sx = pl->Stride_n;
    const int D = t->D, L = t->L, Ms = t->M, Ns = t->N;
    const int Md = PADDING_OUT_LEN(Ms, W, Sy) <= 0 ? 1 : PADDING_OUT_LEN(Ms, W, Sy);
    const int Nd = PADDING_OUT_LEN(Ns, H, Sx) <= 0 ? 1 : PADDING_OUT_LEN(Ms, H, Sx);

    if (!pl->out.data) pl->out = tensor_init(D, Md, Nd, L);

    if (pl->strategy == MAXPOOL)
        tensor_maxpool(t, &pl->out, W, H, Sx, Sy);
    else
        tensor_avgpool(t, &pl->out, W, H, Sx, Sy);
}


void poolLayer_backward(FloatTensor *dout, PoolLayer *pl) {
    exit(-2);
}

void set_pooling_strategy(PoolLayer *pl, int strategy) {
    pl->strategy = strategy;
}