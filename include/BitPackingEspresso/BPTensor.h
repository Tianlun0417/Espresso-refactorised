#ifndef BPTENSOR_H
#define BPTENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "FloatTypeEspresso/Utilities.h"
#include "BPGEMM.h"

typedef struct bit_packed_tensor {
    int D, M, N, L, MNL;
    size_t bytes;
    size_t packed_len;
    __uint32_t *data;
    int packed_by_row;
}BPTensor;

BPTensor bp_tensor_init(int D, int M, int N, int L, int packed_by_row);

BPTensor bp_tensor_zeros(int D, int M, int N, int L, int packed_by_row);

BPTensor bp_tensor_copy(BPTensor *in);

BPTensor bp_tensor_copy_pad(BPTensor *t, int p);

BPTensor bp_tensor_from_ptr(int D, int M, int N, int L, __uint32_t *ptr);

void bp_tensor_tch(BPTensor *a, BPTensor *b);

void bp_tensor_clear(BPTensor *t);

void bp_tensor_pad(BPTensor *src, BPTensor *dst, int p);

void bp_tensor_maxpool(BPTensor *input, BPTensor *output, int pool_kernel_w, int pool_kernel_h,
                       int Sx, int Sy);
void bp_tensor_avgpool(BPTensor *input, BPTensor *output, int pool_kernel_w, int pool_kernel_h,
                       int Sx, int Sy);

void bp_tensor_lower(const int *input_params, __uint8_t *input, BPTensor *output,
                     int conv_kernel_w, int conv_kernel_h, int Sx, int Sy);

void bp_unpack_to_float(float *arr_float, const __uint32_t *arr_packed, size_t packed_size);

void bp_pack_from_float(const float *arr_float, __uint32_t *arr_packed, size_t packed_size);

void bp_tensor_cat(BPTensor *tensor_a, BPTensor *tensor_b, BPTensor *result, int dim);

void bp_tensor_free(BPTensor *t);

void bp_print_tensor(BPTensor *tensor);

#endif //ESPRESSO_REFACTORISED_BPTENSOR_H
