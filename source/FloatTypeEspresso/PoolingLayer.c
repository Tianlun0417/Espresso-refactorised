#include "FloatTypeEspresso/PoolingLayer.h"


poolLayer poolLayer_init(int M, int N, int Sm, int Sn, poolingStrategy strategy) {
    poolLayer pl = {M, N, Sm, Sn, strategy};
    pl.out.data = NULL;
    pl.mask.data = NULL;
    return pl;
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
        exit(-3);
}


void poolLayer_backward(FloatTensor *dout, poolLayer *pl) {
    exit(-2);
}

void set_pooling_strategy(poolLayer *pl, int strategy) {
    pl->strategy = strategy;
}