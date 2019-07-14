#include "ESP_RE.h"
#include "Cifar10Loader.h"

const char * image_path = "/home/tianlun/codes/espresso-refactorised/data/test_batch.bin";
//void print_tensor(FloatTensor* tensor);


int main(){
    FloatTensor cifar_image = tensor_init(1, CIFAR_IMAGE_W, CIFAR_IMAGE_H, CIFAR_CHANNEL);
    FloatTensor cifar_label = tensor_init(1, 1, 1, 1);

    inputLayer input_layer;

    denseLayer dense_layer_1 = denseLayer_init(1024, 8192);
    denseLayer dense_layer_2 = denseLayer_init(1024, 1024);
    denseLayer dense_layer_3 = denseLayer_init(10, 1024);

    bnormLayer bnorm_layer_1 = bnormLayer_init(0);
    bnormLayer bnorm_layer_2 = bnormLayer_init(0);
    bnormLayer bnorm_layer_3 = bnormLayer_init(0);
    bnormLayer bnorm_layer_4 = bnormLayer_init(0);
    bnormLayer bnorm_layer_5 = bnormLayer_init(0);
    bnormLayer bnorm_layer_6 = bnormLayer_init(0);
    bnormLayer bnorm_layer_7 = bnormLayer_init(0);
    bnormLayer bnorm_layer_8 = bnormLayer_init(0);
    bnormLayer bnorm_layer_9 = bnormLayer_init(0);

    convLayer conv_layer_1 = convLayer_init(1, 1, 1);
    convLayer conv_layer_2 = convLayer_init(1, 1, 1);
    convLayer conv_layer_3 = convLayer_init(1, 1, 1);
    convLayer conv_layer_4 = convLayer_init(1, 1, 1);
    convLayer conv_layer_5 = convLayer_init(1, 1, 1);
    convLayer conv_layer_6 = convLayer_init(1, 1, 1);

    poolLayer pool_layer_1 = poolLayer_init(2, 2, 2, 2);
    poolLayer pool_layer_2 = poolLayer_init(2, 2, 2, 2);
    poolLayer pool_layer_3 = poolLayer_init(2, 2, 2, 2);

    /*------------------------Initialise each layer------------------------*/
    init_dense_layer(&dense_layer_1, 1024, 8192);
    init_dense_layer(&dense_layer_2, 1024, 1024);
    init_dense_layer(&dense_layer_3, 10, 1024);

    init_batchnorm_layer(&bnorm_layer_1, 128);
    init_batchnorm_layer(&bnorm_layer_2, 128);
    init_batchnorm_layer(&bnorm_layer_3, 256);
    init_batchnorm_layer(&bnorm_layer_4, 256);
    init_batchnorm_layer(&bnorm_layer_5, 512);
    init_batchnorm_layer(&bnorm_layer_6, 512);
    init_batchnorm_layer(&bnorm_layer_7, 1024);
    init_batchnorm_layer(&bnorm_layer_8, 1024);
    init_batchnorm_layer(&bnorm_layer_9, 10);

    init_conv_layer(&conv_layer_1, 128, 3, 3, 3);
    init_conv_layer(&conv_layer_2, 128, 3, 3, 128);
    init_conv_layer(&conv_layer_3, 256, 3, 3, 128);
    init_conv_layer(&conv_layer_4, 256, 3, 3, 256);
    init_conv_layer(&conv_layer_5, 512, 3, 3, 256);
    init_conv_layer(&conv_layer_6, 512, 3, 3, 512);

    /*------------------------Initialise each layer------------------------*/

    int save = 1;

    for(int idx=0; idx<TEST_IMG; idx++){
        cifar10_load(image_path, idx, 1, &cifar_image, &cifar_label);

        inputLayer_load(&cifar_image, &input_layer);
        inputLayer_forward(&input_layer);

        convLayer_forward(&input_layer.out, &conv_layer_1, save);
        bnormLayer_forward(&conv_layer_1.out, &bnorm_layer_1, save);
        convLayer_forward(&conv_layer_1.out, &conv_layer_2, save);
        poolLayer_forward(&conv_layer_2.out, &pool_layer_1);
        bnormLayer_forward(&pool_layer_1.out, &bnorm_layer_2, save);

        convLayer_forward(&pool_layer_1.out, &conv_layer_3, save);
        bnormLayer_forward(&conv_layer_3.out, &bnorm_layer_3, save);
        convLayer_forward(&conv_layer_3.out, &conv_layer_4, save);
        poolLayer_forward(&conv_layer_4.out, &pool_layer_2);
        bnormLayer_forward(&pool_layer_2.out, &bnorm_layer_4, save);

        convLayer_forward(&pool_layer_2.out, &conv_layer_5, save);
        bnormLayer_forward(&conv_layer_5.out, &bnorm_layer_5, save);
        convLayer_forward(&conv_layer_5.out, &conv_layer_6, save);
        poolLayer_forward(&conv_layer_6.out, &pool_layer_3);
        bnormLayer_forward(&pool_layer_3.out, &bnorm_layer_6, save);

        denseLayer_forward(&pool_layer_3.out, &dense_layer_1, save);
        bnormLayer_forward(&dense_layer_1.out, &bnorm_layer_7, save);
        denseLayer_forward(&dense_layer_1.out, &dense_layer_2, save);
        bnormLayer_forward(&dense_layer_2.out, &bnorm_layer_8, save);

        denseLayer_forward(&dense_layer_2.out, &dense_layer_3, save);
        bnormLayer_forward(&dense_layer_3.out, &bnorm_layer_9, save);

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
        tensor_free(&conv_layer_6.in);
        tensor_free(&conv_layer_6.out);
        tensor_free(&dense_layer_1.in);
        tensor_free(&dense_layer_1.out);
        tensor_free(&dense_layer_2.in);
        tensor_free(&dense_layer_2.out);
        tensor_free(&dense_layer_3.in);
        tensor_free(&dense_layer_3.out);
    }

    return 0;
}
