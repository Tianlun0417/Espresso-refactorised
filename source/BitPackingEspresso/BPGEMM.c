#include "BitPackingEspresso/BPGEMM.h"

void transpose_matrix(int len_row, int len_col,
        const __uint32_t *input_mat, __uint32_t *output_mat) {
    int mat_idx = 0;
    for (int i = 0; i < len_row; i++){
        for (int j = 0; j < len_col; j++){
            if (mat_idx > len_row * len_col)
                fprintf(stderr, "Matrix Transpose: Index Outbound!\n The index is: %d", mat_idx);
            output_mat[j * len_row + i] = input_mat[i * len_col + j];
            mat_idx++;
        }
    }
}

void bitpacking_gemm(BP_GEMM transpose_a, BP_GEMM transpose_b, int M, int N, int K, const __uint32_t *mat_a, int lda,
                     const __uint32_t *mat_b, int ldb, __uint32_t *mat_c, int ldc) {
    const __uint32_t *tmp_a = mat_a;
    const __uint32_t *tmp_b = mat_b;
    __uint32_t * trans_a;
    __uint32_t * trans_b;
    if (transpose_a == Trans){
        trans_a = malloc(M * K * sizeof(__uint32_t));
        transpose_matrix(ldb, lda, mat_a, trans_a);
        tmp_a = trans_a;
    }
    if (transpose_b == Trans){
        trans_b = malloc(N * K * sizeof(__uint32_t));
        transpose_matrix(ldc, ldb, mat_b, trans_b);
        tmp_b = trans_b;
    }

    int64_t sum = 0;
    // multiplication of bit-packed matrix
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for (int p = 0; p < K; p++){
                __uint32_t temp = ~(tmp_a[i*K+p] ^ tmp_b[p*N+j]);
                sum += ((__builtin_popcount(temp))<<1) - 32;
            }
            mat_c[i*N+j] = sum;
            sum = 0;
        }
    }
    if (transpose_a == Trans) free(trans_a);
    if (transpose_b == Trans) free(trans_b);
}

void print_bits(size_t const size, void const *const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    for (i=size-1;i>=0;i--)
        for (j=7;j>=0;j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    printf("\n");
}
