#include "BitPackingEspresso/BPConvolutionalLayer.h"
#include <cblas.h>

__uint32_t *scratch = NULL;

void bp_conv_layer_init(BPConvLayer *conv_layer_ptr, int L, int D, int M, int N,
        int Stride_m, int Stride_n, int padding) {
    // L - no input channels
    // D - no output channels
    // M - kernel height
    // N - kernel width

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
}

void bp_conv_layer_copy_input(BPTensor *t, BPConvLayer *cl) {
    if (!cl->in.data)
        cl->in = bp_tensor_init(t->D, t->M, t->N, t->L, t->packed_by_row);
    memcpy(cl->in.data, t->data, t->bytes);
}

BPTensor bp_conv_layer_pad_input(BPTensor *t, int *M, int *N, int padding) {
    BPTensor tp;
    *M = PAD(*M, padding);
    *N = PAD(*N, padding);
    tp = bp_tensor_copy_pad(t, padding);

    return tp;
}

void bp_conv_layer_free(BPConvLayer *cl) {
    bp_tensor_free(&cl->W);
    bp_tensor_free(&cl->b);
    bp_tensor_free(&cl->out);
    bp_tensor_free(&cl->in);
}

void bp_conv_layer_forward(BPTensor *input_tensor, BPConvLayer *cl, int save) {
    // D - num output channels
    // M - height
    // N - width
    // L - num input channels

//    puts("The inputs of Conv layer: ");
//    for (int i = 0; i < input_tensor->packed_len; i++)
//        printf("%u, ", input_tensor->data[i]);
//    puts("");

    __uint32_t *scr = scratch;
    BPTensor padded_input, tmp;
    int D = input_tensor->D, Ms = input_tensor->M, Ns = input_tensor->N, Ls = input_tensor->L;
    int F = cl->D, W = cl->M, H = cl->N, L = cl->L;
    int p = cl->padding, Sy = cl->Stride_m, Sx = cl->Stride_n;
    ASSERT(input_tensor->L == cl->L, "err: conv shape\n");

    if (save) bp_conv_layer_copy_input(input_tensor, cl);

//    // unpack the input tensor
//    __uint8_t *unpacked_input = malloc(input_tensor->packed_len * 32 * sizeof(__uint8_t));

    if (p) padded_input = bp_conv_layer_pad_input(input_tensor, &Ms, &Ns, p);

    // lower
    const int Md = LOWER_OUT_LEN(Ms, H, Sy);
    const int Nd = LOWER_OUT_LEN(Ns, W, Sx);

    const int Ld = W * H * L;
    if (!scratch) tmp = bp_tensor_init(D, Md, Nd, Ld, input_tensor->packed_by_row);
    else tmp = bp_tensor_from_ptr(D, Md, Nd, Ld, scr);

    bp_tensor_lower(p ? &padded_input : input_tensor, &tmp, W, H, Sx, Sy);

    // mat mul
    if (!cl->out.data) cl->out = bp_tensor_init(D, Md, Nd, F, 0);
    int M = Md * Nd, N = F, K = cl->W.MNL;

    M /= 32;
    N /= 32;
    if (M == 0) M = 1;
//    M = ceil(M / 32) + 1;
    bitpacking_gemm(NoTrans, Trans, M, N, K, tmp.data,
            K, cl->W.data, K, cl->out.data, N);

//    puts("The outputs of Conv layer: ");
//    for (int i = 0; i < cl->out.packed_len; i++)
//        printf("%u, ", cl->out.data[i]);
//    puts("");

    if (!scratch)
        bp_tensor_free(&tmp);
    if (!scratch && p)
        bp_tensor_free(&padded_input);
}



