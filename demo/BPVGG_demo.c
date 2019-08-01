#include "BitPackingEspresso/BP_ESP.h"
#include "DataLoader/Cifar10Loader.h"
#include "CNN/BPVGG.h"
#include <stdbool.h>

const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";
bool use_batch_norm = true;

int main(){
    int num_classes = 10;
    __uint8_t *image = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

    BPVGG *vgg = malloc(sizeof(BPVGG));
    BPVGG_init(vgg, ConfigA, num_classes, use_batch_norm);

    for (int idx = 0; idx < TEST_IMG; idx++){
        cifar10_load_int(image_path, idx, 1, image, label);
        BPTensor image_tensor = bp_tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
        bp_input_layer_forward(image, &image_tensor);

        BPVGG_forward(&image_tensor, vgg);

        printf("NO.%d ", idx);
        for (int i = 0; i < num_classes; i++)
            printf(" %f", vgg->output[i]);
        puts("");

        bp_tensor_free(&image_tensor);
    }

    free(image);
    free(label);
    BPVGG_free(vgg);
    free(vgg);
}