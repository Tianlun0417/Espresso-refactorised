#include "IntTypeEspresso/IntTensor.h"


IntTensor tensor_init(int D, int M, int N, int L) {
    IntTensor t = {D, M, N, L, M * N * L, BYTES(EspInt, D * M * N * L)};
    t.data = MALLOC(EspInt, D * M * N * L);
    ASSERT(t.data, "err: IntTensor malloc");
    return t;
}

IntTensor tensor_zeros(int D, int M, int N, int L) {
    IntTensor t = tensor_init(D, M, N, L);
    memset(t.data, 0, t.bytes);
    return t;
}

IntTensor tensor_copy(IntTensor *in) {
    const int D = in->D, M = in->M, N = in->N, L = in->L;
    IntTensor out = tensor_init(D, M, N, L);
    ASSERT(in->data, "err: null tens\n");
    memcpy(out.data, in->data, in->bytes);
    return out;
}

IntTensor tensor_copy_pad(IntTensor *t, int p) {
    const int Ms = t->M, Ns = t->N, L = t->L, D = t->D;
    const int Md = PAD(Ms, p), Nd = PAD(Ns, p);
    IntTensor out = tensor_zeros(D, Md, Nd, L);
    EspInt *pin = t->data;
    EspInt *pout = out.data;
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

IntTensor tensor_from_ptr(int D, int M, int N, int L, EspInt *ptr) {
    IntTensor t = {D, M, N, L, M * N * L, BYTES(EspInt, D * M * N * L)};
    ASSERT(ptr, "err: NULL ptr\n");
    t.data = ptr;
    return t;
}

void tensor_tch(IntTensor *a, IntTensor *b) {
    const int M = a->M, N = a->N, L = a->L, D = a->D;
    for (int w = 0; w < D; w++) {
        EspInt *src = a->data + w * a->MNL;
        EspInt *dst = b->data + w * b->MNL;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < L; k++)
                    dst[ID3(j, k, i, L, M)] =
                            src[ID3(i, j, k, N, L)];
    }
}

void tensor_clear(IntTensor *t) {
    memset(t->data, 0, t->bytes);
}

void tensor_pad(IntTensor *src, IntTensor *dst, int p) {
    const int D = src->D, L = src->L;
    const int Ms = src->M, Ns = src->N;
    const int Md = dst->M, Nd = dst->N;
    ASSERT(D == dst->D && L == dst->L, "err: pad shape\n");
    EspInt *s = src->data;
    EspInt *d = dst->data;
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

void tensor_maxpool(IntTensor *src, IntTensor *dst, int W, int H,
                    int Sx, int Sy) {
    const int D = src->D, L = src->L;
    const int Ms = src->M, Ns = src->N;
    const int Md = dst->M, Nd = dst->N;
    ASSERT(D == dst->D && L == dst->L, "err: pool shape\n");
    EspInt *d = dst->data;
    int n = 0;
    for (int w = 0; w < D; w++) {
        EspInt *s = src->data + w * src->MNL;
        for (int i = 0; i < Ms; i += Sy)
            for (int j = 0; j < Ns; j += Sx)
                for (int k = 0; k < L; k++) {
                    EspInt v, max = INT_MAX;
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++) {
                            v = s[ID3(i + y, j + x, k, Ns, L)];
                            if (v > max) max = v;
                        }
                    d[n++] = max;
                }
    }
}

void tensor_lower(IntTensor *src, IntTensor *dst,
                  int W, int H, int Sx, int Sy) {
    const int D = src->D;
    const int Ms = src->M, Ns = src->N, Ls = src->L;
    const int Md = dst->M, Nd = dst->N, Ld = src->L;
    ASSERT(Ls == Ld && dst->D == D, "err: lowering shape\n");
    EspInt *d = dst->data;
    int n = 0;
    for (int w = 0; w < D; w++) {
        EspInt *s = src->data + w * src->MNL;
        for (int i = 0; i < Md; i++)
            for (int j = 0; j < Nd; j++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        for (int k = 0; k < Ld; k++)
                            d[n++] =
                                    s[ID3(i * Sy + y, j * Sx + x, k, Ns, Ls)];
    }
}

void tensor_sign(IntTensor *t) {
    for (int i = 0; i < tensor_len(t); i++)
        t->data[i] = 2 * (t->data[i] > 0) - 1;
}

void tensor_free(IntTensor *t) {
    if (t->data) {
        free(t->data);
        t->data = NULL;
    }
}