#include "FloatTypeEspresso/Cifar10Loader.h"


void cifar10_load(const char *tf, int start, int num,
                  FloatTensor *pixel, FloatTensor *label) {
    ASSERT(start + num <= TEST_IMG, "err: cifar num\n");
    ASSERT(pixel->MNL == CIFAR_IMAGE_SIZE, "err: input shape\n");
    uint8_t X_buff[CIFAR_IMAGE_SIZE];
    uint8_t y_buff;
    FILE *pf = fopen(tf, "rb");
    ASSERT(pf, "err: fopen \n");
    FloatTensor tmpX = tensor_init(num, CIFAR_CHANNEL, CIFAR_IMAGE_W, CIFAR_IMAGE_H);
    tensor_clear(label);
    fseek(pf, (CIFAR_IMAGE_SIZE + 1) * start, SEEK_SET);
    for (int i = 0; i < num; i++) {
        float *outX = tmpX.data + i * tmpX.MNL;
        float *outy = label->data + i * label->MNL;
        fread(&y_buff, sizeof(uint8_t), 1, pf);
        fread(X_buff, sizeof(uint8_t), CIFAR_IMAGE_SIZE, pf);
        outy[i] = y_buff;
        for (int j = 0; j < CIFAR_IMAGE_SIZE; j++)
            outX[j] = (float) X_buff[j];
    }

    tensor_tch(&tmpX, pixel);
    tensor_free(&tmpX);
}

