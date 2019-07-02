#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include "Tensor.h"
#include "Utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

void signAct_forward(FloatTensor *t);
void signAct_backward(FloatTensor *dout);
void reluAct_forward(FloatTensor *t);
void reluAct_backward(FloatTensor *dout);
void softmaxAct_forward(FloatTensor *t);
void softmaxAct_backward(FloatTensor *dout);


#ifdef __cplusplus
}
#endif

#endif //ACTIVATIONFUNCTIONS_H
