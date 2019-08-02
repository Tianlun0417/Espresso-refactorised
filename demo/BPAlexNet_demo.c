#include "CNN/BPAlexNet.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main() {
    int num_classes = 10;
    __uint8_t *image = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

    BPAlexNet *alex_net = malloc(sizeof(BPAlexNet));
    BPAlexNet_init(alex_net, num_classes);

    for (int idx = 0; idx < TEST_IMG; idx++) {
        cifar10_load_int(image_path, idx, 1, image, label);
        BPTensor image_tensor = bp_tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
        bp_input_layer_forward(image, &image_tensor);

        BPAlexNet_forward(&image_tensor, alex_net);

        printf("NO.%d ", idx);
        for (int i = 0; i < num_classes; i++)
            printf(" %f", alex_net->output[i]);
        puts("");

        bp_tensor_free(&image_tensor);
    }

    free(image);
    free(label);
    BPAlexNet_free(alex_net);
    free(alex_net);
}
