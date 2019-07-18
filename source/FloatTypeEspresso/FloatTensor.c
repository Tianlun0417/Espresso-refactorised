#include "FloatTypeEspresso/FloatTensor.h"
#include "FloatTypeEspresso/Utilities.h"
#include "stdio.h"


FloatTensor tensor_init(int D, int M, int N, int L) {
    FloatTensor t = {D, M, N, L, M * N * L, BYTES(float, D * M * N * L)};
    t.data = MALLOC(float, D * M * N * L);
    ASSERT(t.data, "err: FloatTensor malloc");
    //float * tmp_ptr = t.data + D * M * N * L;
    //printf("The dim of malloc tensor is %d, %d, %d, %d.\n", D, M, N, L);
    //printf("The pointer value is %p. The total size is %d.\n", t.data, D * M * N * L);
    //printf("The end value of allocated memory chunk should be: %p.\n\n", tmp_ptr);

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
        t->data = NULL;
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

void tensor_lower(FloatTensor *src, FloatTensor *dst,
                  int W, int H, int Sx, int Sy) {
    const int D = src->D;
    const int Ms = src->M, Ns = src->N, Ls = src->L;
    const int Md = dst->M, Nd = dst->N, Ld = src->L;
    ASSERT(Ls == Ld && dst->D == D, "err: lowering shape\n");
    float *d = dst->data;
    int n = 0;
    for (int w = 0; w < D; w++) {
        float *s = src->data + w * src->MNL;
        for (int i = 0; i < Md; i++)
            for (int j = 0; j < Nd; j++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        for (int k = 0; k < Ld; k++)
                            d[n++] =
                                    s[ID3(i * Sy + y, j * Sx + x, k, Ns, Ls)];
    }
}


void tensor_maxpool(FloatTensor *src, FloatTensor *dst, int W, int H,
                    int Sx, int Sy) {
    const int D = src->D, L = src->L;
    const int Ms = src->M, Ns = src->N;
    const int Md = dst->M, Nd = dst->N;
    ASSERT(D == dst->D && L == dst->L, "err: pool shape\n");
    float *d = dst->data;
    int n = 0;
    for (int w = 0; w < D; w++) {
        float *s = src->data + w * src->MNL;
        for (int i = 0; i < Ms; i += Sy)
            for (int j = 0; j < Ns; j += Sx)
                for (int k = 0; k < L; k++) {
                    float v, max = FLT_MIN;
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++) {
                            v = s[ID3(i + y, j + x, k, Ns, L)];
                            if (v > max) max = v;
                        }
                    d[n++] = max;
                }
    }
}


void tensor_avgpool(FloatTensor *src, FloatTensor *dst, int W, int H,
                    int Sx, int Sy){
    const int D = src->D, L = src->L;
    const int Ms = src->M, Ns = src->N;
    const int Md = dst->M, Nd = dst->N;
    ASSERT(D == dst->D && L == dst->L, "err: pool shape\n");
    float *d = dst->data;
    int n = 0;
    for (int w = 0; w < D; w++) {
        float *s = src->data + w * src->MNL;
        for (int i = 0; i < Ms; i += Sy)
            for (int j = 0; j < Ns; j += Sx)
                for (int k = 0; k < L; k++) {
                    float sum = 0;
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++) {
                            sum += s[ID3(i + y, j + x, k, Ns, L)];
                        }
                    d[n++] = sum / (H * W);
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

FloatTensor *tensor_cat(FloatTensor *tensor_a, FloatTensor *tensor_b, int dim) {
    FloatTensor *result = (FloatTensor*) malloc(sizeof(FloatTensor));

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
        result->MNL = tensor_a->MNL;
    }

    result->bytes = BYTES(float, result->D * result->M * result->N * result->L);
    result->data = (float*) malloc(result->bytes);

    for(int idx = 0; idx < tensor_len(tensor_a); idx++){
        result->data[idx] = tensor_a->data[idx];
    }
    for(int idx = 0; idx < tensor_len(tensor_b); idx++){
        result->data[tensor_len(tensor_a) + idx] = tensor_a->data[idx];
    }

    return result;
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