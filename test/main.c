#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ESP_RE.h"

#define THRESHOLD 0.1
const char * image_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-images-idx3-ubyte";
const char * label_path = "/home/tianlun/codes/espresso-refactorised/data/t10k-labels-idx1-ubyte";

int main() {
    /*---------------------Initialise the layers----------------------*/
    inputLayer input_layer;
    denseLayer dense_layer_1 = denseLayer_init(4096, 784);
    denseLayer dense_layer_2 = denseLayer_init(4096, 4096);
    denseLayer output_layer  = denseLayer_init(4096, 10);
    /*---------------------Initialise the layers----------------------*/

    /*-------Random initialise the weights for the dense layers-------*/
    // The weights of dense layers
    float static arr_weight_1[3211264];          // 784 * 4096
    float static arr_weight_2[16777216];         // 4096 * 4096
    float static arr_weight_output_layer[40960]; // 10 * 4096

    for(int i=0; i<3211264; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_weight_1[i] = 1;
        else arr_weight_1[i] = 0;
    }

    for(int i=0; i<16777216; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_weight_2[i] = 1;
        else arr_weight_2[i] = 0;
    }

    for(int i=0; i<40960; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_weight_output_layer[i] = 1;
        else arr_weight_output_layer[i] = 0;
    }

    // The tensors for batch normalisation layer
    float static arr_bnorm_mean[4096];
    float static arr_bnorm_istd[4096];
    float static arr_bnorm_gamma[4096];
    float static arr_bnorm_beta[4096];

    for(int i=0; i<4096; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_mean[i] = 1;
        else arr_bnorm_mean[i] = 0;
    }

    for(int i=0; i<4096; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_istd[i] = 1;
        else arr_bnorm_istd[i] = 0;
    }

    for(int i=0; i<4096; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_gamma[i] = 1;
        else arr_bnorm_gamma[i] = 0;
    }

    for(int i=0; i<4096; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_beta[i] = 1;
        else arr_bnorm_beta[i] = 0;
    }

    float arr_bnorm_output_mean[10];
    float arr_bnorm_output_istd[10];
    float arr_bnorm_output_gamma[10];
    float arr_bnorm_output_beta[10];

    for(int i=0; i<10; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_output_mean[i] = 1;
        else arr_bnorm_output_mean[i] = 0;
    }

    for(int i=0; i<10; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_output_istd[i] = 1;
        else arr_bnorm_output_istd[i] = 0;
    }

    for(int i=0; i<10; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_output_gamma[i] = 1;
        else arr_bnorm_output_gamma[i] = 0;
    }

    for(int i=0; i<10; i++){
        if((float) rand()/(float) (RAND_MAX)>THRESHOLD) arr_bnorm_output_beta[i] = 1;
        else arr_bnorm_output_beta[i] = 0;
    }
    /*-------Random initialise the weights for the dense layers-------*/

    /*-----------------------Load MNIST dataset-----------------------*/
    mnist_dataset_t * mnist_dataset = mnist_get_dataset(image_path, label_path);
    float image_tmp[784];
    for(int i=0; i<784; i++){
        image_tmp[i] = mnist_dataset->images->pixels[i];
    }

    floatTensors mnist_image_tensor = ftens_from_ptr(1, 28, 28, 1, &image_tmp[0]);
    /*-----------------------Load MNIST dataset-----------------------*/

    /*---------Load the weights and input data for each layer---------*/
    inputLayer_load(&mnist_image_tensor, &input_layer);

    floatTensors dense_weight_1 = ftens_from_ptr(1, 4096, 784, 1, &arr_weight_1[0]);
    denseLayer_set(&dense_weight_1, &dense_layer_1);

    floatTensors dense_weight_2 = ftens_from_ptr(1, 4096, 4096, 1, &arr_weight_2[0]);
    denseLayer_set(&dense_weight_2, &dense_layer_2);

    floatTensors output_layer_weight = ftens_from_ptr(1, 10, 4096, 1, &arr_weight_output_layer[0]);
    denseLayer_set(&output_layer_weight, &output_layer);

    bnormLayer batch_norm_layer = bnormLayer_init(0);
    floatTensors bnorm_mean = ftens_from_ptr(1, 4096, 1, 1, &arr_bnorm_mean[0]);
    floatTensors bnorm_istd = ftens_from_ptr(1, 4096, 1, 1, &arr_bnorm_beta[0]);
    floatTensors bnorm_gamma = ftens_from_ptr(1, 4096, 1, 1, &arr_bnorm_gamma[0]);
    floatTensors bnorm_beta = ftens_from_ptr(1, 4096, 1, 1, &arr_bnorm_beta[0]);
    bnormLayer_set(&bnorm_mean, &bnorm_istd, &bnorm_gamma, &bnorm_beta, &batch_norm_layer);

    bnormLayer output_batch_norm_layer = bnormLayer_init(0);
    floatTensors output_bnorm_mean = ftens_from_ptr(1, 10, 1, 1, &arr_bnorm_output_mean[0]);
    floatTensors output_bnorm_istd = ftens_from_ptr(1, 10, 1, 1, &arr_bnorm_output_beta[0]);
    floatTensors output_bnorm_gamma = ftens_from_ptr(1, 10, 1, 1, &arr_bnorm_output_gamma[0]);
    floatTensors output_bnorm_beta = ftens_from_ptr(1, 10, 1, 1, &arr_bnorm_output_beta[0]);
    bnormLayer_set(&output_bnorm_mean, &output_bnorm_istd, &output_bnorm_gamma,
            &output_bnorm_beta, &output_batch_norm_layer);
    /*---------Load the weights and input data for each layer---------*/

    /*-------------Forward passing through the network-------------*/
    int save = 1;
    inputLayer_forward(&input_layer);

    denseLayer_forward(&input_layer.out, &dense_layer_1, save);
    bnormLayer_forward(&dense_layer_1.out, &batch_norm_layer, save);

    denseLayer_forward(&dense_layer_1.out, &dense_layer_2, save);
    bnormLayer_forward(&dense_layer_2.out, &batch_norm_layer, save);

    denseLayer_forward(&dense_layer_2.out, &output_layer, save);
    bnormLayer_forward(&output_layer.out, &output_batch_norm_layer, save);
    /*-------------Forward passing through the network-------------*/

    float results[10];
    for(int i=0; i<10; i++){
        //printf("%f\n", *(output_layer.out.data));
        //results[i] = *(output_layer.out.data);
        //output_layer.out.data++;

        printf("%f\n", *(output_batch_norm_layer.in.data));
        results[i] = *(output_batch_norm_layer.in.data);
        output_batch_norm_layer.in.data++;
    }

    float temp = 0.0;
    int predicted_num = 0;
    for(int i=0; i<10; i++){
        if(i == 0 && results[0] < 0) temp = results[0];
        if(results[i] > temp){
            temp = results[i];
            predicted_num = i;
        }
    }
    printf("the predicted number is %d and the value is %f \n",
           predicted_num, results[predicted_num]);

    return 0;
}