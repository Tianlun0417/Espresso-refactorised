#include "CNN/AlexNet.h"
#include "DataLoader/Cifar10Loader.h"


const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main() {
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    InputLayer input_layer;

    AlexNet *alex_net = malloc(sizeof(AlexNet));
    AlexNet_init(alex_net, 10);
    
    for (int idx = 0; idx < 10; idx++) {
        cifar10_load(image_path, idx, 1, &cifar_image, &cifar_label);

        input_layer_load(&cifar_image, &input_layer);
        input_layer_forward(&input_layer);

        AlexNet_forward(&(input_layer.out), alex_net);

        printf("NO.%d ", idx);
        print_tensor(&(alex_net->output));
    }

    tensor_free(&cifar_label);
    input_layer_free(&input_layer);
    AlexNet_free(alex_net);
    free(alex_net);
}
