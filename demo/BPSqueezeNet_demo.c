#include "CNN/BPSqueezeNet.h"
#include "BitPackingEspresso/BP_ESP.h"
#include "DataLoader/Cifar10Loader.h"


int main(){
    int num_classes = 10;
    __uint8_t *image = malloc(CIFAR_IMAGE_SIZE * sizeof(__uint8_t));
    __uint8_t *label = malloc(sizeof(__uint8_t));

    BPSqueezeNet *squeeze_net = malloc(sizeof(BPSqueezeNet));
    BPSqueezeNet_init(squeeze_net, Version1_0, num_classes);

    for (int idx = 0; idx < TEST_IMG; idx++) {
        cifar10_load_int(image_path, idx, 1, image, label);
        BPTensor image_tensor = bp_tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
        bp_input_layer_forward(image, &image_tensor);

        BPSqueezeNet_forward(&image_tensor, squeeze_net);

        printf("NO.%d ", idx);
        bp_print_tensor(&(squeeze_net->output));

        bp_tensor_free(&image_tensor);
    }

    free(image);
    free(label);
    BPSqueezeNet_free(squeeze_net);
    free(squeeze_net);
}

