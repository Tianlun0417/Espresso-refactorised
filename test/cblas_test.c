#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>


int main(){
    float arrA[] = {1, 2, 3, 4, 5, 6};  // dim of matrix A is 3 x 2 (m x k)
    float arrB[] = {1, 2, 3, 4, 5, 6};  // dim of matrix B is 2 x 3 (k x n)
    float arrC[9] = {};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                3, 3, 2, 1, arrA, 3, arrB, 3,
                0, arrC, 3);
    for(int i = 0; i < 9; i++){
        printf("%f, ", arrC[i]);
    }
//    float* conv_w_arr  = (float*) malloc(9 * sizeof(float));
//    for(int i = 0; i < 9; i++){
//        printf("%f, ", conv_w_arr[i]);
//    }

    return 0;
}


