#ifndef ESPRESSO_REFACTORISED_POOLINGLAYER_H
#define ESPRESSO_REFACTORISED_POOLINGLAYER_H
#include "FloatTensor.h"
#include "Utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {MAX, AVG} poolingStrategy;

typedef struct {
    int M, N, Stride_m, Stride_n; poolingStrategy op;
    FloatTensor out, mask;
} poolLayer;


poolLayer poolLayer_init(int M, int N, int Sm, int Sn);
void poolLayer_free(poolLayer *pl);
void poolLayer_forward(FloatTensor *t, poolLayer *pl);
void poolLayer_backward(FloatTensor *dout, poolLayer *pl);


#ifdef __cplusplus
}
#endif

#endif //ESPRESSO_REFACTORISED_POOLINGLAYER_H
