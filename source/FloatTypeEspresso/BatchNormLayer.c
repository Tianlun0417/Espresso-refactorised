#include "FloatTypeEspresso/BatchNormLayer.h"
#include "FloatTypeEspresso/Utilities.h"


bnormLayer bnormLayer_init(int use_global) {
    bnormLayer bnl;
    BNORML_INIT(bnl);
    bnl.ug = use_global;
    return bnl;
}

bnormLayer *new_bn_layer(int size) {
    bnormLayer *bn_layer_ptr = (bnormLayer *) malloc(sizeof(bnormLayer));
    bn_layer_ptr->N = 0;
    bn_layer_ptr->ug = 0;
    bn_layer_ptr->mean.data = NULL;
    bn_layer_ptr->istd.data = NULL;
    bn_layer_ptr->beta.data = NULL;
    bn_layer_ptr->gamma.data = NULL;
    bn_layer_ptr->in.data = NULL;

    return bn_layer_ptr;
}

void bnormLayer_free(bnormLayer *bnl) {
    tensor_free(&bnl->mean);
    tensor_free(&bnl->istd);
    tensor_free(&bnl->beta);
    tensor_free(&bnl->gamma);
    tensor_free(&bnl->in);
}

void bnormLayer_print_shape(bnormLayer *bnl) {
    printf("bnorm: %d %d\n", bnl->N, bnl->ug);
}

void bnormLayer_set(FloatTensor *mean, FloatTensor *istd,
                    FloatTensor *gamma, FloatTensor *beta, bnormLayer *bnl) {
    const int N = tensor_len(mean);
    ASSERT(N == tensor_len(istd) &&
           N == tensor_len(beta) &&
           N == tensor_len(gamma), "err: bnorm shape\n");

    bnormLayer_free(bnl);
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

void bnormLayer_forward(FloatTensor *input_tensor, bnormLayer *batchnorm_layer, int save) {
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


void bnormLayer_backward(FloatTensor *dt, bnormLayer *bnl) {
    fprintf(stderr, "not implemented\n");
    exit(-2);
}


void bnormLayer_update(bnormLayer *bnl) {
    fprintf(stderr, "not implemented\n");
    exit(-2);
}
