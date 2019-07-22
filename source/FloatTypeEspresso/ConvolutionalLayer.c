#include <cblas.h>
#include "FloatTypeEspresso/ConvolutionalLayer.h"

float *scratch = NULL;


ConvLayer convLayer_init(int Sm, int Sn, int padding) {
    ConvLayer cl;
    CONVL_INIT(cl);
    cl.Stride_m = Sm;
    cl.Stride_n = Sn;
    cl.padding = padding;
    return cl;
}


void conv_layer_init(ConvLayer *conv_layer_ptr, int L, int D, int M, int N,
                     int Stride_m, int Stride_n, int padding) {
    // L - no input channels
    // D - no output channels
    // M - kernel height
    // N - kernel width

//    conv_layer_ptr = (ConvLayer*) malloc(sizeof(ConvLayer));
    conv_layer_ptr->D = D;
    conv_layer_ptr->M = M;
    conv_layer_ptr->N = N;
    conv_layer_ptr->L = L;
    conv_layer_ptr->Stride_m = Stride_m;
    conv_layer_ptr->Stride_n = Stride_n;
    conv_layer_ptr->padding = padding;
    conv_layer_ptr->W.data = NULL;
    conv_layer_ptr->b.data = NULL;
    conv_layer_ptr->in.data = NULL;
    conv_layer_ptr->out.data = NULL;

//    return conv_layer_ptr;
}


void conv_layer_free(ConvLayer *cl) {
    tensor_free(&cl->W);
    tensor_free(&cl->b);
    tensor_free(&cl->out);
    tensor_free(&cl->in);
}


void convLayer_print_shape(ConvLayer *cl) {
    printf("conv: D=%d M=%d N=%d L=%d Stride_m=%d Stride_n=%d padding=%d\n",
           cl->D, cl->M, cl->N, cl->L, cl->Stride_m, cl->Stride_n, cl->padding);
}


void convLayer_set(FloatTensor *W, ConvLayer *cl) {
    int D = W->D, M = W->M, N = W->N, L = W->L;
    if(cl->W.data != NULL)
        tensor_free(&cl->W);
    cl->D = D;
    cl->M = M;
    cl->N = N;
    cl->L = L;
    cl->W = tensor_copy(W);
}


void convLayer_copy_input(FloatTensor *t, ConvLayer *cl) {
    if (!cl->in.data)
        cl->in = tensor_init(t->D, t->M, t->N, t->L);
    memcpy(cl->in.data, t->data, t->bytes);
}


FloatTensor convLayer_pad_input(FloatTensor *t, float *scr,
                                int *M, int *N, int padding) {
    FloatTensor tp;
    *M = PAD(*M, padding);
    *N = PAD(*N, padding);
    if (!scratch) tp = tensor_copy_pad(t, padding);
    else {
        const int D = t->D, L = t->L;
        tp = tensor_from_ptr(D, *M, *N, L, scr);
        tensor_pad(t, &tp, padding);
        scr += (*M) * (*N) * L * D;
    }

    return tp;
}


void conv_layer_forward(FloatTensor *input_t, ConvLayer *cl, int save) {

    // D - num output channels
    // M - height
    // N - width
    // L - num input channels

    float *scr = scratch;
    FloatTensor padded_input, tmp;
    int D = input_t->D, Ms = input_t->M, Ns = input_t->N, Ls = input_t->L;
    int F = cl->D, W = cl->M, H = cl->N, L = cl->L;
    int p = cl->padding, Sy = cl->Stride_m, Sx = cl->Stride_n;
    ASSERT(input_t->L == cl->L, "err: conv shape\n");

    if (save) convLayer_copy_input(input_t, cl);
    if (p) padded_input = convLayer_pad_input(input_t, scr, &Ms, &Ns, p);

    // lower
    const int Md = LOWER_OUT_LEN(Ms, H, Sy);
    const int Nd = LOWER_OUT_LEN(Ns, W, Sx);

    const int Ld = W * H * L;
    if (!scratch) tmp = tensor_init(D, Md, Nd, Ld);
    else tmp = tensor_from_ptr(D, Md, Nd, Ld, scr);

    tensor_lower(p ? &padded_input : input_t, &tmp, W, H, Sx, Sy);

    // mat mul
    if (!cl->out.data) cl->out = tensor_init(D, Md, Nd, F);
    int M = Md * Nd, N = F, K = cl->W.MNL;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1, tmp.data, K, cl->W.data, K,
                0, cl->out.data, N);

    if (!scratch)
        tensor_free(&tmp);
    if (!scratch && p)
        tensor_free(&padded_input);
}


void convLayer_backward(FloatTensor *dout, ConvLayer *cl) {
    exit(-2);
}


void convLayer_update(ConvLayer *cl) {
    exit(-3);
}


