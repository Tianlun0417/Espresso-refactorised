#include <string.h>
#include <cblas.h>
#include "FloatTypeEspresso/DenseLayer.h"
#include "FloatTypeEspresso/Utilities.h"


DenseLayer denseLayer_init(int M, int N) {
    DenseLayer dl;
    DENSEL_INIT(dl);
    dl.M = M;
    dl.N = N;
    return dl;
}


void dense_layer_init(DenseLayer *dense_layer_ptr, int M, int N) {
    dense_layer_ptr->M = M;
    dense_layer_ptr->N = N;
    dense_layer_ptr->W = tensor_init(1, M, N, 1);
    dense_layer_ptr->b.data = NULL;
    dense_layer_ptr->in.data = NULL;
    dense_layer_ptr->out.data = NULL;
}



void denseLayer_free(DenseLayer *dl) {
    tensor_free(&dl->W);
    tensor_free(&dl->b);
    //ftens_free(&dl->dW);  tensor_free(&dl->db);
    tensor_free(&dl->out);
    tensor_free(&dl->in);
}


void denseLayer_print_shape(DenseLayer *dl) {
    printf("dense: %d %d\n", dl->M, dl->N);
}


void denseLayer_set(FloatTensor *W, DenseLayer *dl) {
    const int M = W->M, N = W->N;
    ASSERT(W->D == 1 && W->L == 1, "err: dense shape\n");
    tensor_free(&dl->W);
    dl->M = M;
    dl->N = N;
    dl->W = tensor_copy(W);
}


void dense_layer_forward(FloatTensor *input_tensor, DenseLayer *dense_layer, int cpy) {
    const int D = input_tensor->D, M = dense_layer->M, N = dense_layer->N;
    ASSERT(input_tensor->MNL == dense_layer->N, "err: dense shape\n");

    if (cpy) {
        int M = input_tensor->M, N = input_tensor->N, L = input_tensor->L;
        if (!dense_layer->in.data) dense_layer->in = tensor_init(D, M, N, L);
        memcpy(dense_layer->in.data, input_tensor->data, input_tensor->bytes);
    }

    if (!dense_layer->out.data) dense_layer->out = tensor_init(D, 1, M, 1);
    const float *a = dense_layer->W.data;
    const float *b = input_tensor->data;
    float *c = dense_layer->out.data;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                D, M, N, 1, b, N, a, N, 0, c, M);
}


void denseLayer_backward(FloatTensor *dout, DenseLayer *dl) {
    fprintf(stderr, "not implemented yet\n");
    exit(-2);
}
