#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct floatTensor {
    int D, M, N, L, MNL;
    int bytes;
    float *data;
} FloatTensor;


FloatTensor tensor_init(int D, int M, int N, int L);

FloatTensor tensor_zeros(int D, int M, int N, int L);
//FloatTensor tensor_ones(int D, int M, int N, int L);
//FloatTensor tensor_rand(int D, int M, int N, int L);
//FloatTensor tensor_rand_range(int D, int M, int N, int L,
//                          float min, float max);

FloatTensor tensor_copy(FloatTensor *in);

FloatTensor tensor_copy_pad(FloatTensor *t, int p);

FloatTensor tensor_from_ptr(int D, int M, int N, int L, float *ptr);
//FloatTensor tensor_from_file(int D, int M, int N, int L, FILE *pf);

void tensor_cat(FloatTensor *tensor_a, FloatTensor *tensor_b, FloatTensor *result, int dim);

//FloatTensor tensor_copy_tch(FloatTensor *a);
void tensor_tch(FloatTensor *a, FloatTensor *b);

void tensor_clear(FloatTensor *t);

//void  tensor_reshape(FloatTensor *t, int D, int M, int N, int L);
void tensor_pad(FloatTensor *src, FloatTensor *dst, int p);

void tensor_maxpool(FloatTensor *input, FloatTensor *output, int pool_kernel_w, int pool_kernel_h,
                    int Sx, int Sy);
void tensor_avgpool(FloatTensor *input, FloatTensor *output, int pool_kernel_w, int pool_kernel_h,
                    int Sx, int Sy);

void tensor_lower(FloatTensor *input, FloatTensor *output,
                  int conv_kernel_w, int conv_kernel_h, int Sx, int Sy);

void tensor_sign(FloatTensor *t);

void tensor_free(FloatTensor *t);
//void tensor_print_shape(FloatTensor *t);
//void tensor_print(FloatTensor *t, const char *fmt);
//void tensor_print_ch(FloatTensor *t, int w, int k, int ii, int jj, const char *fmt);

void print_tensor(FloatTensor *tensor);

static inline
int tensor_len(FloatTensor *t) { return t->bytes / sizeof(float); }


#ifdef __cplusplus
}
#endif

#endif //ESPRESSO_REFACTORISED_TENSOR_H
