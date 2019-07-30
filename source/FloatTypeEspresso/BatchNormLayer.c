#include "FloatTypeEspresso/BatchNormLayer.h"
#include "FloatTypeEspresso/Utilities.h"


BnormLayer bnormLayer_init(int use_global) {
    BnormLayer bnl;
    BNORML_INIT(bnl);
    bnl.ug = use_global;
    return bnl;
}

void bnorm_layer_init(BnormLayer *bn_layer_ptr, size_t size) {
    bn_layer_ptr->N = size;
    bn_layer_ptr->ug = 0;
    bn_layer_ptr->mean = tensor_init(1, 1, bn_layer_ptr->N, 1);
    bn_layer_ptr->istd = tensor_init(1, 1, bn_layer_ptr->N, 1);
    bn_layer_ptr->beta = tensor_init(1, 1, bn_layer_ptr->N, 1);
    bn_layer_ptr->gamma = tensor_init(1, 1, bn_layer_ptr->N, 1);
    bn_layer_ptr->in.data = NULL;
}

void bnorm_layer_free(BnormLayer *bnl) {
    tensor_free(&bnl->mean);
    tensor_free(&bnl->istd);
    tensor_free(&bnl->beta);
    tensor_free(&bnl->gamma);
    tensor_free(&bnl->in);
}

void bnormLayer_print_shape(BnormLayer *bnl) {
    printf("bnorm: %d %d\n", bnl->N, bnl->ug);
}

void bnormLayer_set(FloatTensor *mean, FloatTensor *istd,
                    FloatTensor *gamma, FloatTensor *beta, BnormLayer *bnl) {
    const int N = tensor_len(mean);
    ASSERT(N == tensor_len(istd) &&
           N == tensor_len(beta) &&
           N == tensor_len(gamma), "err: bnorm shape\n");

    bnorm_layer_free(bnl);
    bnl->N = N;
    bnl->mean = tensor_copy(mean);
    bnl->istd = tensor_copy(istd);
    bnl->beta = tensor_copy(beta);
    bnl->gamma = tensor_copy(gamma);
}


static
void bnorm(const float *mean, const float *istd,
           const float *beta, const float *gamma,
           const int len, const int N,
           float *in) {
    for (int i = 0; i < len; i++)
        in[i] = ((in[i] - mean[i % N]) *
                 (istd[i % N] * gamma[i % N]) -
                 (beta[i % N]));
}

void bnorm_layer_forward(FloatTensor *input_tensor, BnormLayer *batchnorm_layer, int save) {
    const int D = input_tensor->D, M = input_tensor->M, N = input_tensor->N, L = input_tensor->L;
    const int asd = L > 1 ? L : N * M;
    ASSERT(asd == batchnorm_layer->N, "err: bnorm shape\n")

    if (save) {
        if (!batchnorm_layer->in.data) batchnorm_layer->in = tensor_init(D, M, N, L);
        memcpy(batchnorm_layer->in.data, input_tensor->data, input_tensor->bytes);
    }

    if (batchnorm_layer->ug) {
        // compute curr mean, istd
        // moving avg -> update globals
        fprintf(stderr, "not implemented\n");
        exit(-3);
    }

    float *in = input_tensor->data;
    float *mean = batchnorm_layer->mean.data;
    float *istd = batchnorm_layer->istd.data;
    float *beta = batchnorm_layer->beta.data;
    float *gamma = batchnorm_layer->gamma.data;

    bnorm(mean, istd, beta, gamma, D * M * N * L, asd, in);
}


void bnormLayer_backward(FloatTensor *dt, BnormLayer *bnl) {
    fprintf(stderr, "not implemented\n");
    exit(-2);
}


void bnormLayer_update(BnormLayer *bnl) {
    fprintf(stderr, "not implemented\n");
    exit(-2);
}
