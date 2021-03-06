#include "FloatTypeEspresso/FLOAT_ESP.h"
#include "DataLoader/Cifar10Loader.h"
#include "CNN/VGG.h"
#include <stdbool.h>

bool use_batch_norm = true;

int main() {
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    InputLayer input_layer;

    VGG *vgg = malloc(sizeof(VGG));
    VGG_init(vgg, ConfigE, 10, use_batch_norm);

    for (int idx = 0; idx < TEST_IMG; idx++){
        cifar10_load(image_path, idx, 1, &cifar_image, &cifar_label);

        input_layer_load(&cifar_image, &input_layer);
        input_layer_forward(&input_layer);

        VGG_forward(&(input_layer.out), vgg);

        printf("NO.%d ", idx);
        print_tensor(&(vgg->output));
    }
    tensor_free(&cifar_label);
    input_layer_free(&input_layer);
    VGG_free(vgg);
    free(vgg);
    return 0;
}
