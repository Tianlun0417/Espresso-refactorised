#include "BatchNormLayer.h"
#include "Utilities.h"


bnormLayer bnormLayer_init(int use_global)
{
    bnormLayer bnl; BNORML_INIT(bnl); bnl.ug = use_global;
    return bnl;
}

void bnormLayer_free(bnormLayer *bnl)
{
    ftens_free(&bnl->mean);  ftens_free(&bnl->istd);
    ftens_free(&bnl->gmean); ftens_free(&bnl->gistd);
    ftens_free(&bnl->beta);  ftens_free(&bnl->gamma);
    ftens_free(&bnl->dbeta); ftens_free(&bnl->dgamma);
    ftens_free(&bnl->tmp);   ftens_free(&bnl->in);
}

void bnormLayer_print_shape(bnormLayer *bnl)
{
    printf("bnorm: %d %d\n", bnl->N, bnl->ug);
}

void bnormLayer_set(floatTensors *mean,  floatTensors *istd,
                    floatTensors *gamma, floatTensors *beta, bnormLayer *bnl)
{
    const int N=ftens_len(mean);
    ASSERT(N == ftens_len(istd) &&
           N == ftens_len(beta) &&
           N == ftens_len(gamma), "err: bnorm shape\n");

    bnormLayer_free(bnl);
    bnl->N     = N;
    bnl->mean  = ftens_copy(mean);
    bnl->istd  = ftens_copy(istd);
    bnl->beta  = ftens_copy(beta);
    bnl->gamma = ftens_copy(gamma);
}


static
void bnorm(const float *mean, const float *istd,
           const float *beta, const float *gamma,
           const int len, const int N,
           float *in)
{
    for (int i=0; i < len; i++)
        in[i] = ((in[i] - mean[i%N]) *
                 (istd[i%N] * gamma[i%N]) -
                 (beta[i%N]));
}

void bnormLayer_forward(floatTensors *input_tensor, bnormLayer *batchnorm_layer, int save)
{
    const int D=input_tensor->D, M=input_tensor->M, N=input_tensor->N, L=input_tensor->L;
    const int asd = L>1 ? L : N*M;
    ASSERT(asd == batchnorm_layer->N, "err: bnorm shape\n")

    if (save) {
        if (!batchnorm_layer->in.data) batchnorm_layer->in=ftens_init(D, M, N, L);
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

    bnorm(mean, istd, beta, gamma, D*M*N*L, asd, in);
}


void bnormLayer_backward(floatTensors *dt, bnormLayer *bnl)
{
    fprintf(stderr, "not implemented\n");
    exit(-2);
}


void bnormLayer_update(bnormLayer *bnl)
{
    fprintf(stderr, "not implemented\n");
    exit(-2);
}
