#include "CNN/BPAlexNet.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main() {
    int num_classes = 10;
    __uint8_t *image = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

    BPInputLayer input_layer;

    BPAlexNet *alex_net = malloc(sizeof(BPAlexNet));
    BPAlexNet_init(alex_net, num_classes);

    for (int idx = 0; idx < 10; idx++) {
        cifar10_load_int(image_path, idx, 1, image, label);

        bp_input_layer_load(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL, image, &input_layer);
        bp_input_layer_forward(&input_layer);

        BPAlexNet_forward(&(input_layer.out), alex_net);

        printf("NO.%d ", idx);
        for (int i = 0; i < num_classes; i++)
            printf(" %f", alex_net->output[i]);
        puts("");
    }

    free(label);
    bp_input_layer_free(&input_layer);
    BPAlexNet_free(alex_net);
    free(alex_net);
}
