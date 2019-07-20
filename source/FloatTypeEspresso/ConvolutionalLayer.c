#include <cblas.h>
#include "FloatTypeEspresso/ConvolutionalLayer.h"

float *scratch = NULL;


convLayer convLayer_init(int Sm, int Sn, int padding) {
    convLayer cl;
    CONVL_INIT(cl);
    cl.Stride_m = Sm;
    cl.Stride_n = Sn;
    cl.padding = padding;
    return cl;
}


convLayer *new_conv_layer(int L, int D, int M, int N,
                          int Stride_m, int Stride_n, int padding) {
    // L - no input channels
    // D - no output channels
    // M - kernel height
    // N - kernel width

    convLayer *conv_layer_ptr = (convLayer *) malloc(sizeof(convLayer));
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

    return conv_layer_ptr;
}


void convLayer_free(convLayer *cl) {
    tensor_free(&cl->W);
    tensor_free(&cl->b);
    //ftens_free(&cl->dW);  tensor_free(&cl->db);
    tensor_free(&cl->out);
    tensor_free(&cl->in);
}


void convLayer_print_shape(convLayer *cl) {
    printf("conv: D=%d M=%d N=%d L=%d Stride_m=%d Stride_n=%d padding=%d\n",
           cl->D, cl->M, cl->N, cl->L, cl->Stride_m, cl->Stride_n, cl->padding);
}


void convLayer_set(FloatTensor *W, convLayer *cl) {
    int D = W->D, M = W->M, N = W->N, L = W->L;
    tensor_free(&cl->W);
    cl->D = D;
    cl->M = M;
    cl->N = N;
    cl->L = L;
    cl->W = tensor_copy(W);
}


void convLayer_copy_input(FloatTensor *t, convLayer *cl) {
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


void convLayer_forward(FloatTensor *input_t, convLayer *cl, int save) {
    float *scr = scratch;
    FloatTensor padded_input, tmp;
    int D = input_t->D, Ms = input_t->M, Ns = input_t->N, Ls = input_t->L;

    // D - num output channels
    // M - height
    // N - width
    // L - num input channels

    int F = cl->D, W = cl->M, H = cl->N, L = cl->L;
    //int F = cl->D, H = cl->M, W = cl->N, L = cl->L;
    int p = cl->padding, Sy = cl->Stride_m, Sx = cl->Stride_n;
    ASSERT(input_t->L == cl->L, "err: conv shape\n");

    if (save) convLayer_copy_input(input_t, cl);
    if (p) padded_input = convLayer_pad_input(input_t, scr, &Ms, &Ns, p);

    // lower
    //const int Md = OUT_LEN(Ms, H, Sy);
    //const int Nd = OUT_LEN(Ns, W, Sx);

    const int Md = OUT_LEN(Ms, H, Sy);
    const int Nd = OUT_LEN(Ns, W, Sx);

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


void convLayer_backward(FloatTensor *dout, convLayer *cl) {
    exit(-2);
}


void convLayer_update(convLayer *cl) {
    exit(-3);
}


void print_tensor(FloatTensor *tensor) {
    int count = 0;
    printf("The shape of the tensor is: %d x %d\n", tensor->M, tensor->N);
    for (int i = 0; i < tensor->M; i++) {
        for (int j = 0; j < tensor->N; j++) {
            printf("%.2f, ", tensor->data[i * tensor->M + j]);
            if (tensor->data[i * tensor->M + j] == -1) count++;
        }
        puts("");
    }
}
