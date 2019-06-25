#include "Tensor.h"
#include "Utilities.h"


floatTensors ftens_init(int D, int M, int N, int L)
{
    floatTensors t = {D, M, N, L, M*N*L, BYTES(float, D*M*N*L)};
    t.data = MALLOC(float, D*M*N*L);
    ASSERT(t.data, "err: floatTensors malloc");
    return t;
}

floatTensors ftens_from_ptr(int D, int M, int N, int L, float *ptr)
{
    floatTensors t = {D, M, N, L, M*N*L, BYTES(float, D*M*N*L)};
    ASSERT(ptr, "err: NULL ptr\n");
    t.data = ptr;
    return t;
}

void ftens_print_shape(floatTensors *t)
{
    printf("floatTensors: %d %d %d %d\n", t->D, t->M, t->N, t->L);
}

void ftens_free(floatTensors *t)
{
    if (t->data) {free(t->data); t->data=NULL;}
}

floatTensors ftens_copy(floatTensors *t)
{
    const int D=t->D, M=t->M, N=t->N, L=t->L;
    floatTensors out = ftens_init(D, M, N, L);
    ASSERT(t->data, "err: null tens\n");
    memcpy(out.data, t->data, t->bytes);
    return out;
}

floatTensors ftens_from_file(int D, int M, int N, int L, FILE *pf)
{
    floatTensors out = ftens_init(D, M, N, L);
    fread(out.data, sizeof(float), D*M*N*L, pf);
    return out;
}

void ftens_reshape(floatTensors *t, int D, int M, int N, int L)
{
    const int len = ftens_len(t);
    ASSERT(len== D*M*N*L, "err: floatTensors reshape\n");
    t->D=D; t->M=M; t->N=N; t->L=L;
}


void ftens_clear(floatTensors *t) {memset(t->data, 0, t->bytes);}

floatTensors ftens_zeros(int D, int M, int N, int L)
{
    floatTensors t = ftens_init(D, M, N, L);
    memset(t.data, 0, t.bytes);
    return t;
}

floatTensors ftens_ones(int D, int M, int N, int L)
{
    floatTensors t = ftens_init(D, M, N, L);
    for (int i=0; i < LEN(t); i++)
        D(t)[i] = 1.0f;
    return t;
}

floatTensors ftens_rand(int D, int M, int N, int L)
{
    floatTensors t = ftens_init(D, M, N, L);
    for (int i=0; i < LEN(t); i++)
        D(t)[i] = (rand() % 255) - 128.0f;
    return t;
}


void ftens_sign(floatTensors *t)
{
    for (int i=0; i < t->bytes/sizeof(float); i++)
        t->data[i] = 2.0f * (t->data[i] > 0.0f) - 1.0f;
}

floatTensors ftens_rand_range(int D, int M, int N, int L,
                       float min, float max)
{
    floatTensors t = ftens_init(D, M, N, L);
    for (int i=0; i < ftens_len(&t); i++)
        t.data[i] = ((max-min)*rand())/RAND_MAX + min;
    return t;
}

floatTensors ftens_copy_tch(floatTensors *a)
{
    const int M=a->M, N=a->N, L=a->L, D=a->D;
    floatTensors b = ftens_init(D, N, L, M);
    for (int w=0; w<D; w++) {
        float *src = a->data + w*a->MNL;
        float *dst = b. data + w*b. MNL;
        for (int i=0; i<M; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<L; k++)
                    dst[ID3(j,k,i,L,M)] =
                            src[ID3(i,j,k,N,L)];
    }
    return b;
}

void ftens_tch(floatTensors *a, floatTensors *b)
{
    const int M=a->M, N=a->N, L=a->L, D=a->D;
    for (int w=0; w<D; w++) {
        float *src = a->data + w*a->MNL;
        float *dst = b->data + w*b->MNL;
        for (int i=0; i<M; i++)
            for (int j=0; j<N; j++)
                for (int k=0; k<L; k++)
                    dst[ID3(j,k,i,L,M)] =
                            src[ID3(i,j,k,N,L)];
    }
}

void ftens_lower(floatTensors *src, floatTensors *dst,
                 int W, int H, int Sx, int Sy)
{
    const int D=src->D;
    const int Ms=src->M, Ns=src->N, Ls=src->L;
    const int Md=dst->M, Nd=dst->N, Ld=src->L;
    ASSERT(Ls == Ld && dst->D == D, "err: lowering shape\n");
    float *d = dst->data; int n=0;
    for (int w=0;  w < D; w++) {
        float *s = src->data + w*src->MNL;
        for (int i=0; i<Md; i++)
            for (int j=0; j<Nd; j++)
                for (int y=0; y<H; y++)
                    for (int x=0; x<W; x++)
                        for (int k=0; k<Ld; k++)
                            d[n++] =
                                    s[ID3(i*Sy+y,j*Sx+x,k,Ns,Ls)];
    }
}


void ftens_maxpool(floatTensors *src, floatTensors *dst, int W, int H,
                   int Sx, int Sy)
{
    const int D =src->D, L =src->L;
    const int Ms=src->M, Ns=src->N;
    const int Md=dst->M, Nd=dst->N;
    ASSERT(D==dst->D && L==dst->L, "err: pool shape\n");
    float *d=dst->data; int n=0;
    for (int w=0; w < D; w++) {
        float *s=src->data + w*src->MNL;
        for (int i=0; i < Ms; i+=Sy)
            for (int j=0; j < Ns; j+=Sx)
                for (int k=0; k < L; k++) {
                    float v, max=FLT_MIN;
                    for (int y=0; y<H; y++)
                        for (int x=0; x<W; x++) {
                            v = s[ID3(i+y,j+x,k,Ns,L)];
                            if (v > max) max = v;
                        }
                    d[n++] = max;
                }
    }
}


floatTensors ftens_copy_pad(floatTensors *t, int p)
{
    const int Ms=t->M, Ns=t->N, L=t->L, D=t->D;
    const int Md=PAD(Ms,p), Nd=PAD(Ns,p);
    floatTensors out = ftens_zeros(D, Md, Nd, L);
    float *pin  = t->data;
    float *pout = out.data;
    for (int w=0; w < D; w++) {
        for (int i=0; i < Ms; i++)
            for (int j=0; j < Ns; j++)
                for (int k=0; k < L; k++)
                    pout[ID3(i+p,j+p,k,Nd,L)] =
                            pin[ID3(i,j,k,Ns,L)];
        pin += t->MNL;
        pout += out.MNL;
    }
    return out;
}

void ftens_pad(floatTensors *src, floatTensors *dst, int p)
{
    const int D=src->D, L=src->L;
    const int Ms=src->M, Ns=src->N;
    const int Md=dst->M, Nd=dst->N;
    ASSERT(D==dst->D && L==dst->L, "err: pad shape\n");
    float *s = src->data;
    float *d = dst->data;
    memset(d, 0, dst->bytes);
    for (int w=0; w < D; w++) {
        for (int i=0; i < Ms; i++)
            for (int j=0; j < Ns; j++)
                for (int k=0; k < L; k++)
                    d[ID3(i+p,j+p,k,Nd,L)] =
                            s[ID3(i,j,k,Ns,L)];
        s += src->MNL;
        d += dst->MNL;
    }
}

void ftens_print(floatTensors *t, const char *fmt)
{
    if (!t->data) {printf("floatTensors NULL\n"); return;}
    const int M=t->M, N=t->N, L=t->L, D=t->D;
    float *ptr = t->data;
    for (int w=0; w < D; w++) {
        for (int i=0; i < M; i++) {
            for (int k=0; k < L; k++) {
                for (int j=0; j < N; j++) {
                    float v = ptr[ID3(i,j,k,N,L)];
                    printf(fmt, v);
                } printf(" | ");
            } NL;
        }
        ptr += t->MNL; NL;
    }
    NL;
}

void ftens_print_ch(floatTensors *t, int w, int k, int ii, int jj,
                    const char *fmt)
{
    if (!t->data) {printf("floatTensors NULL\n"); return;}
    const int D=t->D, M=t->M, N=t->N, L=t->L;
    ASSERT(w < D, "err: print\n");
    float *ptr = t->data + w * t->MNL;
    for (int i=0; i < MIN(M, (unsigned)ii); i++) {
        for (int j=0; j < MIN(N,(unsigned)jj); j++) {
            printf(fmt, ptr[ID3(i,j,k,N,L)]);
        }
        NL;
    }
}