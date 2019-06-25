#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int D, M, N, L, MNL;
    int bytes;
    float *data;
} floatTensors;


floatTensors ftens_init(int D, int M, int N, int L);
floatTensors ftens_zeros(int D, int M, int N, int L);
floatTensors ftens_ones(int D, int M, int N, int L);
floatTensors ftens_rand(int D, int M, int N, int L);
floatTensors ftens_rand_range(int D, int M, int N, int L,
                       float min, float max);

floatTensors ftens_copy(floatTensors *in);
floatTensors ftens_copy_pad(floatTensors *t, int p);

floatTensors ftens_from_ptr(int D, int M, int N, int L, float *ptr);
floatTensors ftens_from_file(int D, int M, int N, int L, FILE *pf);

floatTensors ftens_copy_tch(floatTensors *a);
void  ftens_tch(floatTensors *a, floatTensors *b);
void  ftens_clear(floatTensors *t);
void  ftens_reshape(floatTensors *t, int D, int M, int N, int L);
void  ftens_pad(floatTensors *src, floatTensors *dst, int p);
void  ftens_maxpool(floatTensors *src, floatTensors *dst, int W, int H,
                    int Sx, int Sy);

void ftens_lower(floatTensors *src, floatTensors *dst,
                 int W, int H, int Sx, int Sy);

void ftens_sign(floatTensors *t);
void ftens_free(floatTensors *t);
void ftens_print_shape(floatTensors *t);
void ftens_print(floatTensors *t, const char *fmt);
void ftens_print_ch(floatTensors *t, int w, int k, int ii, int jj, const char *fmt);

static inline
int ftens_len(floatTensors *t) {return t->bytes/sizeof(float);}


#ifdef __cplusplus
}
#endif

#endif //ESPRESSO_REFACTORISED_TENSOR_H
