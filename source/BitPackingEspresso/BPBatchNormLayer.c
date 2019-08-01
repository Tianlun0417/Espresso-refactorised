#include "BitPackingEspresso/BPBatchNormLayer.h"


static
void bnorm(const __uint32_t *mean, const __uint32_t *istd,
           const __uint32_t *beta, const __uint32_t *gamma,
           const int len, const int N, __uint32_t *in) {
    for (int i = 0; i < len; i++)
        in[i] = ((in[i] - mean[i % N]) *
                 (istd[i % N] * gamma[i % N]) -
                 (beta[i % N]));
}

void bp_bnorm_layer_init(BPBnormLayer *bn_layer, size_t size) {
    bn_layer->N = size;
    bn_layer->ug = 0;
    bn_layer->mean = bp_tensor_init(1, 1, bn_layer->N, 1, 1);
    bn_layer->istd = bp_tensor_init(1, 1, bn_layer->N, 1, 1);
    bn_layer->beta = bp_tensor_init(1, 1, bn_layer->N, 1, 1);
    bn_layer->gamma = bp_tensor_init(1, 1, bn_layer->N, 1, 1);
    bn_layer->in.data = NULL;
}

void bp_bnorm_layer_free(BPBnormLayer *bnl) {
    bp_tensor_free(&bnl->mean);
    bp_tensor_free(&bnl->istd);
    bp_tensor_free(&bnl->beta);
    bp_tensor_free(&bnl->gamma);
    bp_tensor_free(&bnl->in);
}

void bp_bnorm_layer_forward(BPTensor *input_tensor, BPBnormLayer *batchnorm_layer, int save) {
    const int D = input_tensor->D, M = input_tensor->M, N = input_tensor->N, L = input_tensor->L;
    const int asd = L > 1 ? L : N * M;
    //ASSERT(asd == batchnorm_layer->N, "err: bnorm shape\n")

    if (save) {
        if (!batchnorm_layer->in.data) batchnorm_layer->in = bp_tensor_init(D, M, N, L, 1);
        memcpy(batchnorm_layer->in.data, input_tensor->data, input_tensor->bytes);
    }

    if (batchnorm_layer->ug) {
        // compute curr mean, istd
        // moving avg -> update globals
        fprintf(stderr, "not implemented\n");
        exit(-3);
    }

    __uint32_t *in = input_tensor->data;
    __uint32_t *mean = batchnorm_layer->mean.data;
    __uint32_t *istd = batchnorm_layer->istd.data;
    __uint32_t *beta = batchnorm_layer->beta.data;
    __uint32_t *gamma = batchnorm_layer->gamma.data;

    bnorm(mean, istd, beta, gamma, input_tensor->packed_len, asd, in);
}


