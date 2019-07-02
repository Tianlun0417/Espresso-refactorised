#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ESP_RE.h"

#define THRESHOLD 0.1
const char * image_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-images-idx3-ubyte";
const char * label_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-labels-idx1-ubyte";

void print_mnist_image(float* image_array);
void random_init_arr(float* arr, size_t arr_length);
void print_tensor(FloatTensor* tensor);

int main() {
    //srand(time(NULL));
    /*---------------------Initialise the layers----------------------*/
    inputLayer input_layer;
    denseLayer dense_layer_1 = denseLayer_init(4096, 784);
    denseLayer dense_layer_2 = denseLayer_init(4096, 4096);
    denseLayer output_layer  = denseLayer_init(4096, 10);
    bnormLayer batch_norm_layer = bnormLayer_init(0);
    bnormLayer output_batch_norm_layer = bnormLayer_init(0);
    /*---------------------Initialise the layers----------------------*/

    /*-----------------------Load MNIST dataset-----------------------*/
    mnist_dataset_t * mnist_dataset = mnist_get_dataset(image_path, label_path);
    /*-----------------------Load MNIST dataset-----------------------*/

    /*-------Random initialise the weights for the dense layers-------*/
    // The weights of dense layers
    static float arr_weight_1[784 * 4096];
    static float arr_weight_2[4096 * 4096];
    static float arr_weight_output_layer[10 * 4096];

    random_init_arr(&arr_weight_1[0], 784 * 4096);
    random_init_arr(&arr_weight_2[0], 4096 * 4096);
    random_init_arr(&arr_weight_output_layer[0], 10 * 4096);

    // The tensors for batch normalisation layer
    static float arr_bnorm_mean[4096];
    static float arr_bnorm_istd[4096];
    static float arr_bnorm_gamma[4096];
    static float arr_bnorm_beta[4096];

    random_init_arr(&arr_bnorm_mean[0], 4096);
    random_init_arr(&arr_bnorm_istd[0], 4096);
    random_init_arr(&arr_bnorm_gamma[0], 4096);
    random_init_arr(&arr_bnorm_beta[0], 4096);

    float arr_bnorm_output_mean[10];
    float arr_bnorm_output_istd[10];
    float arr_bnorm_output_gamma[10];
    float arr_bnorm_output_beta[10];

    random_init_arr(&arr_bnorm_output_mean[0], 10);
    random_init_arr(&arr_bnorm_output_istd[0], 10);
    random_init_arr(&arr_bnorm_output_gamma[0], 10);
    random_init_arr(&arr_bnorm_output_beta[0], 10);

    /*-------Random initialise the weights for the dense layers-------*/

    /*---------Load the weights and input data for each layer---------*/
    FloatTensor dense_weight_1 = tensor_from_ptr(1, 4096, 784, 1, &arr_weight_1[0]);
    denseLayer_set(&dense_weight_1, &dense_layer_1);

    FloatTensor dense_weight_2 = tensor_from_ptr(1, 4096, 4096, 1, &arr_weight_2[0]);
    denseLayer_set(&dense_weight_2, &dense_layer_2);

    FloatTensor output_layer_weight = tensor_from_ptr(1, 10, 4096, 1, &arr_weight_output_layer[0]);
    denseLayer_set(&output_layer_weight, &output_layer);

    FloatTensor bnorm_mean = tensor_from_ptr(1, 4096, 1, 1, &arr_bnorm_mean[0]);
    FloatTensor bnorm_istd = tensor_from_ptr(1, 4096, 1, 1, &arr_bnorm_beta[0]);
    FloatTensor bnorm_gamma = tensor_from_ptr(1, 4096, 1, 1, &arr_bnorm_gamma[0]);
    FloatTensor bnorm_beta = tensor_from_ptr(1, 4096, 1, 1, &arr_bnorm_beta[0]);
    bnormLayer_set(&bnorm_mean, &bnorm_istd, &bnorm_gamma, &bnorm_beta, &batch_norm_layer);

    FloatTensor output_bnorm_mean = tensor_from_ptr(1, 10, 1, 1, &arr_bnorm_output_mean[0]);
    FloatTensor output_bnorm_istd = tensor_from_ptr(1, 10, 1, 1, &arr_bnorm_output_beta[0]);
    FloatTensor output_bnorm_gamma = tensor_from_ptr(1, 10, 1, 1, &arr_bnorm_output_gamma[0]);
    FloatTensor output_bnorm_beta = tensor_from_ptr(1, 10, 1, 1, &arr_bnorm_output_beta[0]);
    bnormLayer_set(&output_bnorm_mean, &output_bnorm_istd, &output_bnorm_gamma,
                   &output_bnorm_beta, &output_batch_norm_layer);
    /*---------Load the weights and input data for each layer---------*/

    int save = 0;
    float correct_predict_count = 0.0;
    for(int loop=0; loop<100; loop++){
        float image_tmp[784];
        for(int i=0; i<784; i++){
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
        print_tensor(&dense_layer_1.out);
        bnormLayer_forward(&dense_layer_1.out, &batch_norm_layer, save);
        signAct_forward(&dense_layer_1.out);
        print_tensor(&dense_layer_1.out);

        denseLayer_forward(&dense_layer_1.out, &dense_layer_2, save);
        print_tensor(&dense_layer_2.out);
        bnormLayer_forward(&dense_layer_2.out, &batch_norm_layer, save);
        signAct_forward(&dense_layer_2.out);
        print_tensor(&dense_layer_2.out);

        denseLayer_forward(&dense_layer_2.out, &output_layer, save);
        print_tensor(&output_layer.out);
        bnormLayer_forward(&output_layer.out, &output_batch_norm_layer, save);
        print_tensor(&output_layer.out);
        /*-------------Forward passing through the network-------------*/

        float results[10];
        for(int i=0; i<10; i++){
            printf("%f\n", output_layer.out.data[i]);
            results[i] = output_layer.out.data[i];
        }

        float temp = results[0];
        int predicted_num = 1;
        for(int i=0; i<10; i++){
            if(results[i] > temp){
                temp = results[i];
                predicted_num = i + 1;
            }
        }
        printf("the predicted number is %d and the value is %f \n",
               predicted_num, results[predicted_num - 1]);
        if(label == predicted_num){
            correct_predict_count+=1;
        }else{
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

    float accuracy = correct_predict_count/100;
    printf("The accuracy is: %f\n", accuracy);

    return 0;
}

void print_mnist_image(float* image_array){
    for(int i=0; i<MNIST_IMAGE_HEIGHT; i++){
        for(int j=0; j<MNIST_IMAGE_WIDTH; j++){
            printf("%.2f ", image_array[i*MNIST_IMAGE_WIDTH + j]);
        }
        puts("");
    }
    puts("");
}

void random_init_arr(float* arr, size_t arr_length){
    for(int i=0; i<arr_length; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr[i] = 1.0f;
        else arr[i] = 0.0f;
    }
}

void print_tensor(FloatTensor* tensor){
    int count = 0;
    for(int i=0; i<tensor->M; i++){
        for(int j=0; j<tensor->N; j++){
            printf("%.2f, ", tensor->data[i*tensor->M + j]);
            if(tensor->data[i*tensor->M + j] == -1) count++;
        }
        puts("");
    }
    printf("The length of tensor is %d. There are %d -1 in the tensor.\n\n",
            tensor->MNL, count);
}