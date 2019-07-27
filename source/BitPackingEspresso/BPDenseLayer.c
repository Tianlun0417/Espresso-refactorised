#include "BitPackingEspresso/BPDenseLayer.h"

void bp_dense_layer_init(BPDenseLayer *dense_layer_ptr, int M, int N) {
    dense_layer_ptr->M = M;
    dense_layer_ptr->N = N;
    dense_layer_ptr->W = bp_tensor_init(1, M, N, 1);
    dense_layer_ptr->b.data = NULL;
    dense_layer_ptr->in.data = NULL;
    dense_layer_ptr->out.data = NULL;
}

void bp_denseLayer_free(BPDenseLayer *dl) {
    bp_tensor_free(&dl->W);
    bp_tensor_free(&dl->b);
    bp_tensor_free(&dl->out);
    bp_tensor_free(&dl->in);
}

void bp_dense_layer_forward(BPTensor *input_tensor, BPDenseLayer *dense_layer, int cpy) {
    const int D = input_tensor->D, N = dense_layer->N;
    int  M = dense_layer->M;
    ASSERT(input_tensor->MNL == dense_layer->N, "err: dense shape\n");

    if (cpy) {
        int M = input_tensor->M, N = input_tensor->N, L = input_tensor->L;
        if (!dense_layer->in.data) dense_layer->in = bp_tensor_init(D, M, N, L);
        memcpy(dense_layer->in.data, input_tensor->data, input_tensor->bytes);
    }

    if (!dense_layer->out.data) dense_layer->out = bp_tensor_init(D, 1, M, 1);
    const __uint32_t *a = dense_layer->W.data;
    const __uint32_t *b = input_tensor->data;
    __uint32_t *c = dense_layer->out.data;

    M /= 32;
    if (M == 0)
        M = 1;
    bitpacking_gemm(NoTrans, Trans, D, M, N, b, N, a, N, c, M);
}
