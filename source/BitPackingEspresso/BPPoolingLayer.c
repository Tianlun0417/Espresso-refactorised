#include "BitPackingEspresso/BPPoolingLayer.h"

void bp_pool_layer_init(BPPoolLayer *pool_layer_ptr, int M, int N, int Stride_m, int Stride_n, int padding,
                        BPPoolingStrategy strategy) {
    pool_layer_ptr->M = M;
    pool_layer_ptr->N = N;
    pool_layer_ptr->Stride_m = Stride_m;
    pool_layer_ptr->Stride_n = Stride_n;
    pool_layer_ptr->strategy = strategy;
    pool_layer_ptr->padding  = padding;
    pool_layer_ptr->out.data = NULL;
    pool_layer_ptr->mask.data = NULL;
}

void bp_pool_layer_free(BPPoolLayer *pl) {
    bp_tensor_free(&pl->out);
    bp_tensor_free(&pl->mask);
}

void bp_pool_layer_forward(BPTensor *t, BPPoolLayer *pl) {
    const int W = pl->M, H = pl->N, Sy = pl->Stride_m, Sx = pl->Stride_n;
    const int D = t->D, L = t->L, Ms = t->M, Ns = t->N;
    const int Md = PADDING_OUT_LEN(Ms, W, Sy) <= 0 ? 1 : PADDING_OUT_LEN(Ms, W, Sy);
    const int Nd = PADDING_OUT_LEN(Ns, H, Sx) <= 0 ? 1 : PADDING_OUT_LEN(Ms, H, Sx);

    if (!pl->out.data) pl->out = bp_tensor_init(D, Md, Nd, L, 0);

    if (pl->strategy == MAXPOOL)
        bp_tensor_maxpool(t, &pl->out, W, H, Sx, Sy);
    else
        bp_tensor_avgpool(t, &pl->out, W, H, Sx, Sy);
}
