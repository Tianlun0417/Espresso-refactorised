#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "FloatTypeEspresso/ESP_RE.h"
#define TEST_SIZE 10000

const char * image_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-images-idx3-ubyte";
const char * label_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-labels-idx1-ubyte";

void print_mnist_image(float* image_array);
void print_tensor(FloatTensor* tensor);

int main() {
    //srand(time(NULL));
    /*---------------------Initialise the layers----------------------*/
    inputLayer input_layer;
    denseLayer dense_layer_1 = denseLayer_init(10, 784);
    bnormLayer output_batch_norm_layer = bnormLayer_init(0);
    /*---------------------Initialise the layers----------------------*/

    /*-----------------------Load MNIST dataset-----------------------*/
    mnist_dataset_t * mnist_dataset = mnist_get_dataset(image_path, label_path);
    /*-----------------------Load MNIST dataset-----------------------*/

    /*----------Random initialise the weights for the layers----------*/
    init_dense_layer(&dense_layer_1, 10, 784);

    // The tensors for batch normalisation layer
    init_batchnorm_layer(&output_batch_norm_layer, 10);
    /*----------Random initialise the weights for the layers----------*/

    int save = 0;
    float correct_predict_count = 0.0;
    for(int loop=0; loop<TEST_SIZE; loop++){
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
        bnormLayer_forward(&dense_layer_1.out, &output_batch_norm_layer, save);
        //print_tensor(&output_layer.out);
        /*-------------Forward passing through the network-------------*/

        float results[10];
        for(int i=0; i<10; i++){
            //printf("%f\n", output_layer.out.data[i]);
            results[i] = dense_layer_1.out.data[i];
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

        for(int aaa = 0; aaa < INT16_MAX; aaa++)
            puts("aaaaaaaaaaaaaaaaaaaaaaaa");
        tensor_free(&input_layer.out);
        tensor_free(&dense_layer_1.out);
        tensor_free(&dense_layer_1.in);
    }
    for(int aaa = 0; aaa < INT32_MAX; aaa++)
        puts("aaaaaaaaaaaaaaaaaaaaaaaa");

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
