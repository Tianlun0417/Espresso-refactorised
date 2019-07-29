//
// Created by tianlun on 26/07/19.
//

#ifndef BPGEMM_H
#define BPGEMM_H

#include <stdio.h>
#include <stdlib.h>

typedef enum bitpacking_gemm{
    Trans, NoTrans
}BP_GEMM;

void transpose_matrix(
        int len_row, int len_col,
        const __uint32_t *input_mat,
        __uint32_t *output_mat);
void bitpacking_gemm(
        BP_GEMM transpose_a, BP_GEMM transpose_b,
        int M, int N, int K,
        const __uint32_t *mat_a, int lda,
        const __uint32_t *mat_b, int ldb,
        __uint32_t *mat_c, int ldc
);

void print_bits(void const * const ptr);

#endif //ESPRESSO_REFACTORISED_BPGEMM_H
