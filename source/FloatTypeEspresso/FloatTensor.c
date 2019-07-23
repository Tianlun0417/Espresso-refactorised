#include "FloatTypeEspresso/FloatTensor.h"
#include "FloatTypeEspresso/Utilities.h"
#include "stdio.h"


FloatTensor tensor_init(int D, int M, int N, int L) {
    FloatTensor t = {D, M, N, L, M * N * L, BYTES(float, D * M * N * L)};
    t.data = MALLOC(float, D * M * N * L);
    ASSERT(t.data, "err: FloatTensor malloc");
    return t;
}

FloatTensor tensor_from_ptr(int D, int M, int N, int L, float *ptr) {
    FloatTensor t = {D, M, N, L, M * N * L, BYTES(float, D * M * N * L)};
    ASSERT(ptr, "err: NULL ptr\n");
    t.data = ptr;
    return t;
}

void tensor_print_shape(FloatTensor *t) {
    printf("FloatTensor: %d %d %d %d\n", t->D, t->M, t->N, t->L);
}

void tensor_free(FloatTensor *t) {
    if (t->data) {
        free(t->data);
    }
}

FloatTensor tensor_copy(FloatTensor *in) {
    const int D = in->D, M = in->M, N = in->N, L = in->L;
    FloatTensor out = tensor_init(D, M, N, L);
    ASSERT(in->data, "err: null tens\n");
    memcpy(out.data, in->data, in->bytes);
    return out;
}

FloatTensor tensor_from_file(int D, int M, int N, int L, FILE *pf) {
    FloatTensor out = tensor_init(D, M, N, L);
    fread(out.data, sizeof(float), D * M * N * L, pf);
    return out;
}

void tensor_reshape(FloatTensor *t, int D, int M, int N, int L) {
    const int len = tensor_len(t);
    ASSERT(len == D * M * N * L, "err: FloatTensor reshape\n");
    t->D = D;
    t->M = M;
    t->N = N;
    t->L = L;
}


void tensor_clear(FloatTensor *t) { memset(t->data, 0, t->bytes); }

FloatTensor tensor_zeros(int D, int M, int N, int L) {
    FloatTensor t = tensor_init(D, M, N, L);
    memset(t.data, 0, t.bytes);
    return t;
}

FloatTensor tensor_ones(int D, int M, int N, int L) {
    FloatTensor t = tensor_init(D, M, N, L);
    for (int i = 0; i < LEN(t); i++)
        D(t)[i] = 1.0f;
    return t;
}

FloatTensor tensor_rand(int D, int M, int N, int L) {
    FloatTensor t = tensor_init(D, M, N, L);
    for (int i = 0; i < LEN(t); i++)
        D(t)[i] = (rand() % 255) - 128.0f;
    return t;
}


void tensor_sign(FloatTensor *t) {
    for (int i = 0; i < t->bytes / sizeof(float); i++)
        t->data[i] = 2.0f * (t->data[i] > 0) - 1.0f;
}

FloatTensor tensor_rand_range(int D, int M, int N, int L,
                              float min, float max) {
    FloatTensor t = tensor_init(D, M, N, L);
    for (int i = 0; i < tensor_len(&t); i++)
        t.data[i] = ((max - min) * rand()) / RAND_MAX + min;
    return t;
}

FloatTensor tensor_copy_tch(FloatTensor *a) {
    const int M = a->M, N = a->N, L = a->L, D = a->D;
    FloatTensor b = tensor_init(D, N, L, M);
    for (int w = 0; w < D; w++) {
        float *src = a->data + w * a->MNL;
        float *dst = b.data + w * b.MNL;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < L; k++)
                    dst[ID3(j, k, i, L, M)] =
                            src[ID3(i, j, k, N, L)];
    }
    return b;
}

void tensor_tch(FloatTensor *a, FloatTensor *b) {
    const int M = a->M, N = a->N, L = a->L, D = a->D;
    for (int w = 0; w < D; w++) {
        float *src = a->data + w * a->MNL;
        float *dst = b->data + w * b->MNL;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < L; k++)
                    dst[ID3(j, k, i, L, M)] =
                            src[ID3(i, j, k, N, L)];
    }
}

void tensor_lower(FloatTensor *input, FloatTensor *output,
                  int conv_kernel_w, int conv_kernel_h, int Sx, int Sy) {
    const int D = input->D;
    const int Ms = input->M, Ns = input->N, Ls = input->L;
    const int Md = output->M, Nd = output->N, Ld = output->L;
    //ASSERT(Ls == Ld && output->D == D, "err: lowering shape\n");
    ASSERT(output->D == D, "err: lowering shape\n")
    float *d = output->data;
    int n = 0;
    int outbound_count = 0, loop_count = 0;
    for (int w = 0; w < D; w++) {
        float *s = input->data + w * input->MNL;
        for (int i = 0; i < Md; i++)
            for (int j = 0; j < Nd; j++)
                for (int y = 0; y < conv_kernel_h; y++)
                    for (int x = 0; x < conv_kernel_w; x++)
                        for (int k = 0; k < Ls; k++){
                            int id3 = ID3(i * Sy + y, j * Sx + x, k, Ns, Ls);
                            loop_count++;
                            if(id3 >= (input->MNL * input->D)) {
                                outbound_count++;
                                fprintf(stderr, "Source Tensor Lower: Index outbund!\n");
                            }
                            d[n] = s[ID3(i * Sy + y, j * Sx + x, k, Ns, Ls)];
                            if (n >= (output->MNL * output->D)) {
                                outbound_count++;
                                fprintf(stderr, "Destiny Tensor Lower: Index outbund!\n");
                            }
                            n++;
                        }

    }
}


void tensor_maxpool(FloatTensor *input, FloatTensor *output, int pool_kernel_w, int pool_kernel_h,
                    int Sx, int Sy) {
    int batch = input->D, kernel_h = input->M, kernel_w = input->N, channel = input->L;
    int output_h = output->M, output_w = output->N;
    int channel_size = input->MNL;
    int input_kernel_size = kernel_h * kernel_w;
    int n = 0;
    int outbound_count = 0, loop_count = 0;

    if (input_kernel_size == 1){
        for (int i = 0; i < output->MNL; i++)
            output->data[i] = input->data[i];
    }else{
        for (int idx_b = 0; idx_b < batch; idx_b++){
            for (int idx_c = 0; idx_c < channel; idx_c++){
                int kernel_loc = idx_b * channel_size + idx_c * input_kernel_size;
                for (size_t output_idx_h = 0; output_idx_h < output_h; output_idx_h++){
                    for (size_t output_idx_w = 0; output_idx_w < output_w; output_idx_w++){
                        float value, max = FLT_MIN;
                        for (size_t i = 0; i < pool_kernel_h; i++){
                            const size_t s = output_idx_h * Sy + i;
                            if (s < kernel_h){
                                for (size_t j = 0; j < pool_kernel_w; j++){
                                    const size_t t = output_idx_w * Sx + j;
                                    loop_count++;
                                    if(kernel_loc + s + t >= (input->MNL * input->D)){
                                        outbound_count++;
                                        int xxx = kernel_loc + s + t;
                                        fprintf(stderr, "Input Maxpool: Index outbund!\n");
                                    }
                                    if (t < kernel_w){
                                        value = input->data[kernel_loc + s + t];
                                        if (value > max) max = value;
                                    }
                                }
                            }
                        }
                        output->data[n] = max;
                        if (n >= (output->MNL * output->D)) {
                            outbound_count++;
                            fprintf(stderr, "Output Maxpool: Index outbund!\n");
                        }
                        n++;
                    }
                }
            }
        }
    }
}

void tensor_avgpool(FloatTensor *input, FloatTensor *output, int pool_kernel_w, int pool_kernel_h,
                    int Sx, int Sy){
    int pool_kernel_size = pool_kernel_w * pool_kernel_h;
    int batch = input->D, kernel_h = input->M, kernel_w = input->N, channel = input->L;
    int output_h = output->M, output_w = output->N;
    int channel_size = input->MNL;
    int input_kernel_size = kernel_h * kernel_w;
    int n = 0;
    int outbound_count = 0, loop_count = 0;

    if (input_kernel_size == 1){
        for (int i = 0; i < output->MNL; i++)
            output->data[i] = input->data[i];
    }else{
        for (int idx_b = 0; idx_b < batch; idx_b++){
            for (int idx_c = 0; idx_c < channel; idx_c++){
                int kernel_loc = idx_b * channel_size + idx_c * input_kernel_size;
                for (size_t output_idx_h = 0; output_idx_h < output_h; output_idx_h++){
                    for (size_t output_idx_w = 0; output_idx_w < output_w; output_idx_w++){
                        float sum = 0;
                        for (size_t i = 0; i < pool_kernel_h; i++){
                            const size_t s = output_idx_h * Sy + i;
                            if (s < kernel_h){
                                for (size_t j = 0; j < pool_kernel_w; j++){
                                    const size_t t = output_idx_w * Sx + j;
                                    loop_count++;
                                    if(kernel_loc + s + t >= (input->MNL * input->D)){
                                        outbound_count++;
                                        fprintf(stderr, "Avgpool: Index outbund!\n");
                                    }
                                    if (t < kernel_w){
                                        sum += input->data[kernel_loc + s + t];
                                    }
                                }
                            }
                        }
                        output->data[n] = sum / pool_kernel_size;
                        n++;
                    }
                }
            }
        }
    }
}

FloatTensor tensor_copy_pad(FloatTensor *t, int p) {
    const int Ms = t->M, Ns = t->N, L = t->L, D = t->D;
    const int Md = PAD(Ms, p), Nd = PAD(Ns, p);
    FloatTensor out = tensor_zeros(D, Md, Nd, L);
    float *pin = t->data;
    float *pout = out.data;
    for (int w = 0; w < D; w++) {
        for (int i = 0; i < Ms; i++)
            for (int j = 0; j < Ns; j++)
                for (int k = 0; k < L; k++)
                    pout[ID3(i + p, j + p, k, Nd, L)] =
                            pin[ID3(i, j, k, Ns, L)];
        pin += t->MNL;
        pout += out.MNL;
    }
    return out;
}

void tensor_pad(FloatTensor *src, FloatTensor *dst, int p) {
    const int D = src->D, L = src->L;
    const int Ms = src->M, Ns = src->N;
    const int Md = dst->M, Nd = dst->N;
    ASSERT(D == dst->D && L == dst->L, "err: pad shape\n");
    float *s = src->data;
    float *d = dst->data;
    memset(d, 0, dst->bytes);
    for (int w = 0; w < D; w++) {
        for (int i = 0; i < Ms; i++)
            for (int j = 0; j < Ns; j++)
                for (int k = 0; k < L; k++)
                    d[ID3(i + p, j + p, k, Nd, L)] =
                            s[ID3(i, j, k, Ns, L)];
        s += src->MNL;
        d += dst->MNL;
    }
}

void tensor_cat(FloatTensor *tensor_a, FloatTensor *tensor_b, FloatTensor *result, int dim) {
    if(dim == 0){
        if((tensor_a->M != tensor_b->M)
        || (tensor_a->N != tensor_b->N)
        || (tensor_a->L != tensor_b->L))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D + tensor_b->D;
        result->M = tensor_a->M;
        result->N = tensor_a->N;
        result->L = tensor_a->L;
        result->MNL = tensor_a->MNL;
    }else if(dim == 1){
        if((tensor_a->D != tensor_b->D)
        || (tensor_a->N != tensor_b->N)
        || (tensor_a->L != tensor_b->L))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D;
        result->M = tensor_a->M + tensor_b->M;
        result->N = tensor_a->N;
        result->L = tensor_a->L;
        result->MNL = tensor_a->MNL;
    }else if(dim == 2){
        if((tensor_a->D != tensor_b->D)
        || (tensor_a->M != tensor_b->M)
        || (tensor_a->L != tensor_b->L))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D;
        result->M = tensor_a->M;
        result->N = tensor_a->N + tensor_b->N;
        result->L = tensor_a->L;
        result->MNL = tensor_a->MNL;
    }else if(dim == 3){
        if((tensor_a->D != tensor_b->D)
        || (tensor_a->N != tensor_b->N)
        || (tensor_a->M != tensor_b->M))
            fprintf(stderr, "Dimension doesn't match\n");
        result->D = tensor_a->D;
        result->M = tensor_a->M;
        result->N = tensor_a->N;
        result->L = tensor_a->L + tensor_b->L;
        result->MNL = result->M * result->N * result->L;
    }

    result->bytes = BYTES(float, result->D * result->M * result->N * result->L);
    result->data  = malloc(result->bytes);

    for(int idx = 0; idx < tensor_len(tensor_a); idx++){
        result->data[idx] = tensor_a->data[idx];
    }
    for(int idx = 0; idx < tensor_len(tensor_b); idx++){
        result->data[tensor_len(tensor_a) + idx] = tensor_a->data[idx];
    }
}

void tensor_print(FloatTensor *t, const char *fmt) {
    if (!t->data) {
        printf("FloatTensor NULL\n");
        return;
    }
    const int M = t->M, N = t->N, L = t->L, D = t->D;
    float *ptr = t->data;
    for (int w = 0; w < D; w++) {
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < L; k++) {
                for (int j = 0; j < N; j++) {
                    float v = ptr[ID3(i, j, k, N, L)];
                    printf(fmt, v);
                }
                printf(" | ");
            }
            NL;
        }
        ptr += t->MNL;
        NL;
    }
    NL;
}

void tensor_print_ch(FloatTensor *t, int w, int k, int ii, int jj,
                     const char *fmt) {
    if (!t->data) {
        printf("FloatTensor NULL\n");
        return;
    }
    const int D = t->D, M = t->M, N = t->N, L = t->L;
    ASSERT(w < D, "err: print\n");
    float *ptr = t->data + w * t->MNL;
    for (int i = 0; i < MIN(M, (unsigned) ii); i++) {
        for (int j = 0; j < MIN(N, (unsigned) jj); j++) {
            printf(fmt, ptr[ID3(i, j, k, N, L)]);
        }
        NL;
    }
}

void print_tensor(FloatTensor *tensor) {
    printf("The shape of the tensor is: %d x %d x %d x %d\n", tensor->D, tensor->M, tensor->N, tensor->L);
    for (int ch = 0; ch < tensor->L; ch++){
        for (int i = 0; i < tensor->M; i++) {
            for (int j = 0; j < tensor->N; j++) {
                printf("%.2f, ", tensor->data[ch * tensor->M * tensor->N + i * tensor->M + j]);
            }
            puts("");
        }
    }
}