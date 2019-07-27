#include "BitPackingEspresso/BPDenseOutputLayer.h"


void bp_dense_output_layer_init(BPDenseOutputLayer *dense_layer_ptr, int M, int N) {
    dense_layer_ptr->output_dim = M;
    dense_layer_ptr->input_dim = N;
    dense_layer_ptr->W = bp_tensor_init(1, M, N, 1);
    dense_layer_ptr->in.data = NULL;
    dense_layer_ptr->output_arr = dense_layer_ptr->output_arr = malloc(M * sizeof(float));;
}

void bp_dense_output_layer_free(BPDenseOutputLayer *dl) {
    bp_tensor_free(&dl->W);
    bp_tensor_free(&dl->in);
//    free(dl->output_arr);
}

void bp_dense_output_layer_forward(BPTensor *input_tensor, BPDenseOutputLayer *dense_layer, int cpy) {
    const int D = input_tensor->D, M = dense_layer->output_dim, N = dense_layer->input_dim;
    ASSERT(input_tensor->MNL == dense_layer->input_dim, "err: dense shape\n");

    if (cpy) {
        int M = input_tensor->M, N = input_tensor->N, L = input_tensor->L;
        if (!dense_layer->in.data) dense_layer->in = bp_tensor_init(D, M, N, L);
        memcpy(dense_layer->in.data, input_tensor->data, input_tensor->bytes);
    }

//    if (dense_layer->output_arr == NULL)
//    dense_layer->output_arr = malloc(D * M * sizeof(float));
    float *a = malloc(dense_layer->W.packed_len * 32 * sizeof(float));
    bp_unpack_to_float(a, dense_layer->W.data, dense_layer->W.packed_len);
    float *b = malloc(input_tensor->packed_len * 32 * sizeof(float));
    bp_unpack_to_float(b, input_tensor->data, input_tensor->packed_len);
    float *c = dense_layer->output_arr;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                D, M, N, 1, b, N, a, N, 0, c, M);

    free(a);
    free(b);
}


