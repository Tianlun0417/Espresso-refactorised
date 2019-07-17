#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "FloatTypeEspresso/FLOAT_ESP.h"

#define TEST_SIZE 5

const char *image_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-images-idx3-ubyte";
const char *label_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-labels-idx1-ubyte";

void print_mnist_image(float *image_array);

int main() {
    /*---------------------Initialise the layers----------------------*/
    inputLayer input_layer;
    denseLayer dense_layer_1 = denseLayer_init(4096, 784);
    denseLayer dense_layer_2 = denseLayer_init(4096, 4096);
    denseLayer output_layer = denseLayer_init(4096, 10);
    bnormLayer batch_norm_layer_1 = bnormLayer_init(0);
    bnormLayer batch_norm_layer_2 = bnormLayer_init(0);
    bnormLayer output_batch_norm_layer = bnormLayer_init(0);
    /*---------------------Initialise the layers----------------------*/

    /*-----------------------Load MNIST dataset-----------------------*/
    mnist_dataset_t *mnist_dataset = mnist_get_dataset(image_path, label_path);
    /*-----------------------Load MNIST dataset-----------------------*/

    /*----------Random initialise the weights for the layers----------*/
    init_dense_layer(&dense_layer_1, 4096, 784);
    init_dense_layer(&dense_layer_2, 4096, 4096);
    init_dense_layer(&output_layer, 10, 4096);

    // The tensors for batch normalisation layer
    init_batchnorm_layer(&batch_norm_layer_1, 4096);
    init_batchnorm_layer(&batch_norm_layer_2, 4096);
    init_batchnorm_layer(&output_batch_norm_layer, 10);
    /*----------Random initialise the weights for the layers----------*/

    int save = 0;
    float correct_predict_count = 0.0;
    for (int loop = 0; loop < TEST_SIZE; loop++) {
        float image_tmp[784];
        for (int i = 0; i < 784; i++) {
            image_tmp[i] = mnist_dataset->images->pixels[i];
        }

        uint8_t label = *(mnist_dataset->labels);
        //print_mnist_image(&image_tmp[0]);
        printf("The actual digit should be %u.\n", label);
        mnist_dataset->images++;
        mnist_dataset->labels++;

        /*-------------Forward passing through the network-------------*/
        FloatTensor mnist_image_tensor = tensor_from_ptr(1, 28, 28, 1, &image_tmp[0]);
        inputLayer_load(&mnist_image_tensor, &input_layer);
        inputLayer_forward(&input_layer);

        denseLayer_forward(&input_layer.out, &dense_layer_1, save);
        //print_tensor(&dense_layer_1.out);
        bnormLayer_forward(&dense_layer_1.out, &batch_norm_layer_1, save);
        //signAct_forward(&dense_layer_1.out);
        //print_tensor(&dense_layer_1.out);

        denseLayer_forward(&dense_layer_1.out, &dense_layer_2, save);
        //print_tensor(&dense_layer_2.out);
        bnormLayer_forward(&dense_layer_2.out, &batch_norm_layer_2, save);
        //signAct_forward(&dense_layer_2.out);
        //print_tensor(&dense_layer_2.out);

        denseLayer_forward(&dense_layer_2.out, &output_layer, save);
        //print_tensor(&output_layer.out);
        bnormLayer_forward(&output_layer.out, &output_batch_norm_layer, save);
        //print_tensor(&output_layer.out);
        /*-------------Forward passing through the network-------------*/

        float results[10];
        for (int i = 0; i < 10; i++) {
            printf("%f\n", output_layer.out.data[i]);
            results[i] = output_layer.out.data[i];
        }

        float temp = results[0];
        int predicted_num = 1;
        for (int i = 0; i < 10; i++) {
            if (results[i] < temp) {
                temp = results[i];
                predicted_num = i + 1;
            }
        }
        printf("\nThe predicted number is %d and the value is %f \n",
               predicted_num, results[predicted_num - 1]);
        if (label == predicted_num) {
            correct_predict_count += 1;
        } else {
            puts("Wrong predict\n");
        }

        tensor_free(&input_layer.out);
        tensor_free(&dense_layer_1.out);
        tensor_free(&dense_layer_1.in);
        tensor_free(&dense_layer_2.out);
        tensor_free(&dense_layer_2.in);
        tensor_free(&output_layer.out);
        tensor_free(&output_layer.in);
    }

//    float accuracy = correct_predict_count/TEST_SIZE;
//    printf("The accuracy is: %f\n", accuracy);

    return 0;
}

void print_mnist_image(float *image_array) {
    for (int i = 0; i < MNIST_IMAGE_HEIGHT; i++) {
        for (int j = 0; j < MNIST_IMAGE_WIDTH; j++) {
            printf("%.2f ", image_array[i * MNIST_IMAGE_WIDTH + j]);
        }
        puts("");
    }
    puts("");
}
