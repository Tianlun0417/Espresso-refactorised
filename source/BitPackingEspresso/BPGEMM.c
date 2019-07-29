#include "BitPackingEspresso/BPGEMM.h"

void transpose_matrix(int len_row, int len_col,
        const __uint32_t *input_mat, __uint32_t *output_mat) {
    // len_row: the size of row before transpose
    // len_col: the size of column before transpose
    int mat_idx = 0;
    for (int i = 0; i < len_row; i++){
        for (int j = 0; j < len_col; j++){
//            if (mat_idx >= len_row * len_col)
//                fprintf(stderr, "Matrix Transpose: Index Outbound!\n The index is: %d", mat_idx);
            __uint32_t aaa = input_mat[i * len_col + j];
            __uint32_t bbb = output_mat[j * len_row + i];
            output_mat[j * len_row + i] = input_mat[i * len_col + j];
            mat_idx++;
        }
    }
}

void bitpacking_gemm(BP_GEMM transpose_a, BP_GEMM transpose_b, int M, int N, int K,
        const __uint32_t *mat_a, int lda, const __uint32_t *mat_b, int ldb,
        __uint32_t *mat_c, int ldc) {
//    const __uint32_t *tmp_a = mat_a;
//    const __uint32_t *tmp_b = mat_b;
    __uint32_t * trans_a;
    __uint32_t * trans_b;

//    puts("BPGEMM matrix A: ");
//    for (int i = 0; i < K; i++) {
//        for (int j = 0; j < M; j++) {
//            printf("%u, ", mat_a[i * M + j]);
//        }
//        puts("");
//    }
//    puts("");
//    puts("BPGEMM matrix B: ");
//    for (int i = 0; i < K; i++) {
//        for (int j = 0; j < N; j++) {
//            printf("%u, ", mat_b[i * N + j]);
//        }
//        puts("");
//    }
//    puts("");

    if (transpose_a == Trans){
        trans_a = malloc(M * K * sizeof(__uint32_t));
        transpose_matrix(lda, K, mat_a, trans_a);
        mat_a = trans_a;

        puts("BPGEMM matrix A (Transposed): ");
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                printf("%u, ", mat_a[i * M + j]);
            }
            puts("");
        }
        puts("");
    }
    if (transpose_b == Trans){

        trans_b = malloc(N * K * sizeof(__uint32_t));
        transpose_matrix(ldb, N, mat_b, trans_b);
        mat_b = trans_b;

//        puts("BPGEMM matrix B (Transposed): ");
//        for (int i = 0; i < N; i++) {
//            for (int j = 0; j < K; j++) {
//                printf("%u, ", mat_b[i * K + j]);
//            }
//            puts("");
//        }
//        puts("");
    }

//    puts("\nBPGEMM: ");
//    for (int i = 0; i < M * K; i++)
//        printf("%u, ", mat_a[i]);
//    puts("");
    // multiplication of bit-packed matrix


//    for (int i = 0; i < M; i++){
//        for (int j = 0; j < N; j++){
//            for (int p = 0; p < K; p++){
//                if (i*K+p >= M * K)
//                    fprintf(stderr, "Matrix A: Index Outbound! Index: %d.\n", i*K+p);
//                if (p*N+j >= K * N)
//                    fprintf(stderr, "Matrix B: Index Outbound! Index: %d.\n", p*N+j);
//                __uint32_t aaa = mat_a[i*K+p];
//                __uint32_t bbb = mat_b[p*N+j];
//                __uint32_t temp = ~(aaa ^ bbb);
////                __uint32_t temp = ~(tmp_a[c_row_idx*K+p] ^ tmp_b[p*N+c_col_idx]);
//                mat_c[i*N+j] += ((__builtin_popcount(temp))<<1) - 32;
//            }
//        }
//    }

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for (int p = 0; p < K; p++){
                if (i*K+p >= M * K)
                    fprintf(stderr, "Matrix A: Index Outbound! Index: %d.\n", i*K+p);
                if (p*N+j >= K * N)
                    fprintf(stderr, "Matrix B: Index Outbound! Index: %d.\n", p*N+j);
                __uint32_t aaa = mat_a[i*K+p];
                __uint32_t bbb = mat_b[p*N+j];
                __uint32_t temp = ~(aaa ^ bbb);
//                __uint32_t temp = ~(tmp_a[c_row_idx*K+p] ^ tmp_b[p*N+c_col_idx]);
                mat_c[i*N+j] += ((__builtin_popcount(temp))<<1) - 32;
            }
        }
    }


    if (transpose_a == Trans) free(trans_a);
    if (transpose_b == Trans) free(trans_b);
}

void print_bits(void const * const ptr){
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    for (i=31;i>=0;i--){
        byte = (b[i] >> i) & 1;
        printf("%u,", byte);
    }

//        for (j=7;j>=0;j--)
//        {
//            byte = (b[i] >> j) & 1;
//            printf("%u,", byte);
//        }
    printf(" | ");
}
