#include "FloatTypeEspresso/ActivationFunctions.h"


void reluAct_forward(FloatTensor *t) {
    const int len = tensor_len(t);
    for (int i = 0; i < len; i++)
        t->data[i] = MAX(0, t->data[i]);
}

void reluAct_backward(FloatTensor *dout) {
    fprintf(stderr, "not implemeted yer\n");
    exit(-4);
}

void signAct_forward(FloatTensor *t) {
    tensor_sign(t);
    //const int len = ftens_len(t);
    //for (int i=0; i < len; i++)
    //t->data[i] = 2.0f * (t->data[i] > 0.0f) - 1.0f;
}


void signAct_backward(FloatTensor *t) {
    fprintf(stderr, "not implemeted yet\n");
    exit(-4);
}

void softmaxAct_forward(FloatTensor *t) {
    fprintf(stderr, "not implemeted yet\n");
    exit(-4);
}


void softmaxAct_backward(FloatTensor *dout) {
    fprintf(stderr, "not implemeted yet\n");
    exit(-4);
}

