#include "FloatTypeEspresso/ESP_RE.h"
#include "FloatTypeEspresso/Cifar10Loader.h"


const char * image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";

int main(){
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    inputLayer input_layer;

    convLayer conv_layer_1 = convLayer_init(4, 4, 2); // stride x, stride y, padding length
    convLayer conv_layer_2 = convLayer_init(1, 1, 2);
    convLayer conv_layer_3 = convLayer_init(1, 1, 1);
    convLayer conv_layer_4 = convLayer_init(1, 1, 1);
    convLayer conv_layer_5 = convLayer_init(1, 1, 1);

    poolLayer maxPool_layer = poolLayer_init(3, 3, 2, 2); // kernel x, kernel y, stride x, stride y

    denseLayer dense_layer_1 = denseLayer_init(4096, 1024);
    denseLayer dense_layer_2 = denseLayer_init(4096, 4096);
    denseLayer dense_layer_3 = denseLayer_init(10, 4096);

    /*------------------------Initialise each layer------------------------*/
    init_conv_layer(&conv_layer_1, 64, 11, 11, 3);
    init_conv_layer(&conv_layer_2, 192, 5, 5, 64);
    init_conv_layer(&conv_layer_3, 384, 3, 3, 192);
    init_conv_layer(&conv_layer_4, 256, 3, 3, 384);
    init_conv_layer(&conv_layer_5, 256, 3, 3, 256);

    init_dense_layer(&dense_layer_1, 4096, 1024);
    init_dense_layer(&dense_layer_2, 4096, 4096);
    init_dense_layer(&dense_layer_3, 10, 4096);
    /*------------------------Initialise each layer------------------------*/

    int save = 1;

    for(int idx=0; idx<TEST_IMG; idx++){
        cifar10_load(image_path, idx, 1, &cifar_image, &cifar_label);

        inputLayer_load(&cifar_image, &input_layer);
        inputLayer_forward(&input_layer);

        convLayer_forward(&input_layer.out, &conv_layer_1, save);
        reluAct_forward(&conv_layer_1.out);
        poolLayer_forward(&conv_layer_1.out, &maxPool_layer);

        convLayer_forward(&maxPool_layer.out, &conv_layer_2, save);
        tensor_free(&maxPool_layer.out);
        reluAct_forward(&conv_layer_2.out);
        poolLayer_forward(&conv_layer_2.out, &maxPool_layer);

        convLayer_forward(&maxPool_layer.out, &conv_layer_3, save);
        tensor_free(&maxPool_layer.out);
        reluAct_forward(&conv_layer_3.out);
        convLayer_forward(&conv_layer_3.out, &conv_layer_4, save);
        reluAct_forward(&conv_layer_4.out);
        convLayer_forward(&conv_layer_4.out, &conv_layer_5, save);
        reluAct_forward(&conv_layer_5.out);
        poolLayer_forward(&conv_layer_5.out, &maxPool_layer);

        denseLayer_forward(&maxPool_layer.out, &dense_layer_1, save);
        reluAct_forward(&dense_layer_1.out);
        denseLayer_forward(&dense_layer_1.out, &dense_layer_2, save);
        reluAct_forward(&dense_layer_2.out);

        denseLayer_forward(&dense_layer_2.out, &dense_layer_3, save);

        printf("Test img NO.%d ", idx);
        print_tensor(&dense_layer_3.out);

        tensor_free(&input_layer.out);

        tensor_free(&conv_layer_1.in);
        tensor_free(&conv_layer_1.out);
        tensor_free(&conv_layer_2.in);
        tensor_free(&conv_layer_2.out);
        tensor_free(&conv_layer_3.in);
        tensor_free(&conv_layer_3.out);
        tensor_free(&conv_layer_4.in);
        tensor_free(&conv_layer_4.out);
        tensor_free(&conv_layer_5.in);
        tensor_free(&conv_layer_5.out);

        tensor_free(&dense_layer_1.in);
        tensor_free(&dense_layer_1.out);
        tensor_free(&dense_layer_2.in);
        tensor_free(&dense_layer_2.out);
        tensor_free(&dense_layer_3.in);
        tensor_free(&dense_layer_3.out);

        tensor_free(&maxPool_layer.out);
    }

    return 0;
}

