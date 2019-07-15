#ifndef INTTENSOR_H
#define INTTENSOR_H

#include "IntTypeEspresso/Utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INT_MAX INT16_MAX
typedef int EspInt;

typedef struct intTensor{
    int D, M, N, L, MNL;
    int bytes;
    EspInt *data;
} IntTensor;


IntTensor tensor_init(int D, int M, int N, int L);
IntTensor tensor_zeros(int D, int M, int N, int L);
IntTensor tensor_copy(IntTensor *in);
IntTensor tensor_copy_pad(IntTensor *t, int p);
IntTensor tensor_from_ptr(int D, int M, int N, int L, EspInt *ptr);
void  tensor_tch(IntTensor *a, IntTensor *b);
void  tensor_clear(IntTensor *t);
void  tensor_pad(IntTensor *src, IntTensor *dst, int p);
void  tensor_maxpool(IntTensor *src, IntTensor *dst, int W, int H,
                     int Sx, int Sy);
void tensor_lower(IntTensor *src, IntTensor *dst,
                  int W, int H, int Sx, int Sy);
void tensor_sign(IntTensor *t);
void tensor_free(IntTensor *t);

static inline
int tensor_len(IntTensor *t) {return t->bytes/sizeof(EspInt);}

#ifdef __cplusplus
}
#endif
#endif //ESPRESSO_REFACTORISED_INTTENSOR_H
