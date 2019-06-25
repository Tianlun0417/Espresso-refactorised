#include <string.h>
#include <cblas.h>
#include "DenseLayer.h"
#include "Utilities.h"


denseLayer denseLayer_init(int M, int N)
{
    denseLayer dl; DENSEL_INIT(dl); dl.M=M; dl.N=N;
    return dl;
}


void denseLayer_free(denseLayer *dl)
{
    ftens_free(&dl->W);   ftens_free(&dl->b);
    //ftens_free(&dl->dW);  ftens_free(&dl->db);
    ftens_free(&dl->out); ftens_free(&dl->in);
}


void denseLayer_print_shape(denseLayer *dl)
{
    printf("dense: %d %d\n", dl->M, dl->N);
}


void denseLayer_set(floatTensors *W, denseLayer *dl)
{
    const int M=W->M, N=W->N;
    ASSERT(W->D==1 && W->L==1, "err: dense shape\n");
    ftens_free(&dl->W);
    dl->M = M; dl->N = N;
    dl->W = ftens_copy(W);
}


void denseLayer_forward(floatTensors *input_tensor, denseLayer *dense_layer, int save)
{
    const int D=input_tensor->D, M=dense_layer->M, N=dense_layer->N;
    ASSERT(input_tensor->MNL == dense_layer->N,  "err: dense shape\n");

    if (save) {
        int M=input_tensor->M, N=input_tensor->N, L=input_tensor->L;
        if (!dense_layer->in.data) dense_layer->in = ftens_init(D,M,N,L);
        memcpy(dense_layer->in.data, input_tensor->data, input_tensor->bytes);
    }

    if (!dense_layer->out.data) dense_layer->out = ftens_init(D, 1, M, 1);
    const float *a=dense_layer->W.data;
    const float *b=input_tensor->data;
    float       *c=dense_layer->out.data;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                D, M, N, 1, b, N, a, N, 0, c, M);

}


void denseLayer_backward(floatTensors *dout, denseLayer *dl)
{
    fprintf(stderr, "not implemented yet\n");
    exit(-2);
}
