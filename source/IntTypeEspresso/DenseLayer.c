#include <string.h>
#include <cblas.h>
#include "IntTypeEspresso/DenseLayer.h"
#include "IntTypeEspresso/Utilities.h"


DenseLayer denseLayer_init(int M, int N) {
    DenseLayer dl;
    DENSEL_INIT(dl);
    dl.M = M;
    dl.N = N;
    return dl;
}


void denseLayer_free(DenseLayer *dl) {
    tensor_free(&dl->W);
    tensor_free(&dl->b);
    tensor_free(&dl->out);
    tensor_free(&dl->in);
}


void denseLayer_print_shape(DenseLayer *dl) {
    printf("dense: %d %d\n", dl->M, dl->N);
}


void denseLayer_set(IntTensor *W, DenseLayer *dl) {
    const int M = W->M, N = W->N;
    ASSERT(W->D == 1 && W->L == 1, "err: dense shape\n");
    tensor_free(&dl->W);
    dl->M = M;
    dl->N = N;
    dl->W = tensor_copy(W);
}


void denseLayer_forward(IntTensor *input_tensor, DenseLayer *dense_layer, int save) {
    const int D = input_tensor->D, M = dense_layer->M, N = dense_layer->N;
    ASSERT(input_tensor->MNL == dense_layer->N, "err: dense shape\n");

    if (save) {
        int M = input_tensor->M, N = input_tensor->N, L = input_tensor->L;
        if (!dense_layer->in.data) dense_layer->in = tensor_init(D, M, N, L);
        memcpy(dense_layer->in.data, input_tensor->data, input_tensor->bytes);
    }

    if (!dense_layer->out.data) dense_layer->out = tensor_init(D, 1, M, 1);
    const EspInt *a = dense_layer->W.data;
    const EspInt *b = input_tensor->data;
    EspInt *c = dense_layer->out.data;
}
