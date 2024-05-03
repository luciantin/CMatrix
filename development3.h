#ifndef CMATRIX_DEVELOPMENT_H
#define CMATRIX_DEVELOPMENT_H

#if DEBUG
#include <stdio.h>
#endif

#include <malloc.h>
#include "helpers/nnc_matrix.h"
//#include "training/AutoGenTest.h"
#include "helpers/nnc_layer.h"
#include "helpers/nnc_optimizer.h"
#include "helpers/nnc_activation_layer.h"
#include "helpers/nnc_loss_function.h"
#include "helpers/nnc_vector.h"
#include "helpers/nnc_config.h"
#include "helpers/nnc_importer.h"

void RunDevelopment(){

    nnc_init_rand(111111);

    #if DEBUG
    setbuf(stdout, NULL); // Clion dprintf lag fix
    #endif

    dprintf("Start\n");
    dputs("--------------");

    NNCIMatrixType inputs_training = NNCImportMatrixFromFile("C:\\Repos\\CMatrix\\training\\datasets\\dataset_numbers_100_train.matrix");
    NNCIMatrixType target_training = NNCImportMatrixFromFile("C:\\Repos\\CMatrix\\training\\datasets\\dataset_numbers_100_truth_train.matrix");

    NNCIMatrixType inputs_test = NNCImportMatrixFromFile("C:\\Repos\\CMatrix\\training\\datasets\\dataset_numbers_100_test.matrix");
    NNCIMatrixType target_test = NNCImportMatrixFromFile("C:\\Repos\\CMatrix\\training\\datasets\\dataset_numbers_100_truth_test.matrix");

    NNCIMatrixType input = inputs_training;
    NNCIMatrixType target = target_training;

    int sample_len = input->y;
    int input_len = input->x;
    int output_len = 10;
    int epoch_len = 1000;

    nnc_mtype momentum = 0;
    nnc_mtype learning_rate = 0.05;
    nnc_mtype decay = 1e-5;
    nnc_mtype epislon = 1e-7;
    nnc_mtype beta_1 = 0.9;
    nnc_mtype beta_2 = 0.999;
    nnc_mtype dropout_rate = 0.2;

    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(input_len, 32);
    NNCIDenseLayerType dense2 = NNCDenseLayerAlloc(32, 24);
    NNCIDenseLayerType dense3 = NNCDenseLayerAlloc(24, output_len);
    NNCDenseLayerSetRegularizationParameters(dense1, 0, 5e-4, 0, 5e-4);

    NNCIOptimizerAdamType optimizerAdam = NNCOptimizerAdamAlloc(learning_rate, decay, epislon, beta_1, beta_2);

    //---------FORWARD PASS---------
    epoch_len += 1;
    for(int epoch = 1; epoch <= epoch_len; epoch ++){

        if(epoch == epoch_len){
            input = inputs_test;
            target = target_test;
            sample_len = input->y;
            input_len = input->x;
            dprintf("Test pass : \n");
        }

        NNCIMatrixType dense1_forward = NNCDenseLayerForward(input, dense1);
        NNCIMatrixType relu1_forward = NNCActivationReLUForward(dense1_forward);

        NNCIMatrixType dense2_forward = NNCDenseLayerForward(relu1_forward, dense2);
        NNCIMatrixType relu2_forward = NNCActivationReLUForward(dense2_forward);

        NNCIMatrixType dense3_forward = NNCDenseLayerForward(relu2_forward, dense3);
        NNCIMatrixType softmax1_forward = NNCActivationSoftMaxForward(dense3_forward);

        NNCIMatrixType ccel1_forward = NNCLossCCELForward(softmax1_forward, target);

        //---------STATISTICS---------
        if(epoch % 1 == 0){
            nnc_mtype mean = NNCMatrixMean(ccel1_forward);
            nnc_vector argmax_prediction = NNCMatrixArgMax(softmax1_forward);
            nnc_vector argmax_target = NNCMatrixToVector(target, 1);
            nnc_mtype regularization_loss = NNCDenseLayerCalculateRegularizationLoss(dense1) + NNCDenseLayerCalculateRegularizationLoss(dense2);

            dprintf("epoch : %d ", epoch);
            dprintf(" lrate : %.9g ", optimizerAdam->current_learning_rate);
            dprintf("acc : %.9g", NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len));
            dprintf(" reg loss : %.6g", regularization_loss);
            dprintf(" mean loss : %.9g \n", mean);

            if(epoch == epoch_len) {
                NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len);
                dprintf("Target :    ");
                NNCVectorPrint(argmax_target, sample_len);
                dprintf("Prediction :");
                NNCVectorPrint(argmax_prediction, sample_len);
                break; // Test epoch
            }

            free(argmax_target);
            free(argmax_prediction);
        }


        //---------BACKWARD PASS---------


        NNCIMatrixType softmax_ccel_backward = NNCActivationSoftMaxLossCCELBackward(softmax1_forward, target);
        NNCDenseLayerBackward(softmax_ccel_backward, dense3);

        NNCIMatrixType relu2_backward = NNCActivationReLUBackward(relu2_forward, dense3->dinputs);
        NNCDenseLayerWithRegularizationBackward(relu2_backward, dense2);

        NNCIMatrixType relu1_backward = NNCActivationReLUBackward(relu1_forward, dense2->dinputs);
        NNCDenseLayerWithRegularizationBackward(relu1_backward, dense1);

        NNCOptimizerAdamPreUpdateParams(optimizerAdam);
        NNCOptimizerAdamUpdateParams(optimizerAdam, dense1);
        NNCOptimizerAdamUpdateParams(optimizerAdam, dense2);
        NNCOptimizerAdamUpdateParams(optimizerAdam, dense3);
        NNCOptimizerAdamPostUpdateParams(optimizerAdam);


        NNCMatrixDeAlloc(dense1_forward);
        NNCMatrixDeAlloc(dense2_forward);
        NNCMatrixDeAlloc(dense3_forward);
        NNCMatrixDeAlloc(relu1_forward);
        NNCMatrixDeAlloc(relu2_forward);
        NNCMatrixDeAlloc(softmax1_forward);
        NNCMatrixDeAlloc(ccel1_forward);
        NNCMatrixDeAlloc(softmax_ccel_backward);
        NNCMatrixDeAlloc(relu1_backward);
        NNCMatrixDeAlloc(relu2_backward);

    }

    dprintf("End\n");

    NNCMatrixDeAlloc(inputs_training);
    NNCMatrixDeAlloc(target_training);
    NNCMatrixDeAlloc(inputs_test);
    NNCMatrixDeAlloc(target_test);
    NNCDenseLayerDeAlloc(dense1);
    NNCDenseLayerDeAlloc(dense2);
    NNCDenseLayerDeAlloc(dense3);
    NNCOptimizerAdamDeAlloc(optimizerAdam);

}

#endif //CMATRIX_DEVELOPMENT_H
