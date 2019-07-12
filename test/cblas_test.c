#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>


int main(){
    const int M=4;
    const int N=2;
    const int K=3;
    const float alpha=1;
    const float beta=0;
    const int lda=M;
    const int ldb=K;
    const int ldc=N;

    float A[12]={1, 2, 3,
                 4, 5, 6,
                 7, 8, 9,
                 8, 7, 6}; // K * M

    float B[6]={5, 4,
                3, 2,
                1, 0}; // N * K

    float C[8];

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            printf("%f, ", C[i*N+j]);
        }
        puts("");
    }

    return 0;
}


