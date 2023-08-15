#ifndef CMATRIX_DEVELOPMENT_H
#define CMATRIX_DEVELOPMENT_H

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include "helpers/nnc_matrix.h"
#include "AutoGenTest.h"
#include "helpers/nnc_layer.h"
#include "helpers/nnc_optimizer.h"
#include "helpers/nnc_activation_layer.h"
#include "helpers/nnc_loss_function.h"
#include "helpers/nnc_vector.h"

void RunDevelopment(){
    srand(111111);

    printf("Start\n");
    puts("--------------");

    NNCIMatrixType inputs_training = GetAutoGenTrainingMatrix();
    NNCIMatrixType target_training = GetAutoGenTrainingTruthMatrix();

    NNCIMatrixType inputs_test = GetAutoGenTestMatrix();
    NNCIMatrixType target_test = GetAutoGenTestTruthMatrix();

    NNCIMatrixType input = inputs_training;
    NNCIMatrixType target = target_training;

    int sample_len = input->y;
    int input_len = input->x;
    int output_len = 3;
    int epoch_len = 100;

    nnc_mtype momentum = 0;
    nnc_mtype learning_rate = 0.05;
    nnc_mtype decay = 1e-5;
    nnc_mtype epislon = 1e-7;
    nnc_mtype beta_1 = 0.9;
    nnc_mtype beta_2 = 0.999;
    nnc_mtype dropout_rate = 0.2;

    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(input_len, 64);
    NNCIDropoutLayerType dropout1 = NNCDropoutLayerAlloc(dropout_rate);
    NNCIDenseLayerType dense2 = NNCDenseLayerAlloc(64, output_len);
    NNCDenseLayerSetRegularizationParameters(dense1, 0, 5e-4, 0, 5e-4);

//    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(input_len, 64);
//    NNCIDenseLayerType dense2 = NNCDenseLayerAlloc(64, 128);
//    NNCIDenseLayerType dense3 = NNCDenseLayerAlloc(128, 64);
//    NNCIDenseLayerType dense4 = NNCDenseLayerAlloc(64, output_len);

    NNCIOptimizerSGDType optimizerSgd = NNCOptimizerSGDAlloc(learning_rate, decay, momentum);
    NNCIOptimizerAdaGradType optimizerAdaGrad = NNCOptimizerAdaGradAlloc(learning_rate, decay);
    NNCIOptimizerRMSPropType optimizerRmsProp = NNCOptimizerRMSPropAlloc(learning_rate, decay);
    NNCIOptimizerAdamType optimizerAdam = NNCOptimizerAdamAlloc(learning_rate, decay, epislon, beta_1, beta_2);

    //---------FORWARD PASS---------
//    epoch_len += 1;
    for(int epoch = 1; epoch <= epoch_len; epoch ++){

//        if(epoch == epoch_len){
//            input = inputs_test;
//            target = target_test;
//            sample_len = input->y;
//            input_len = input->x;
//            printf("Test pass : ");
//        }

        NNCIMatrixType dense1_forward = NNCDenseLayerForward(input, dense1);
        NNCIMatrixType relu1_forward = NNCActivationReLUForward(dense1_forward);

        NNCIMatrixType dropout1_forward = NNCDropoutLayerForward(relu1_forward, dropout1);
//        NNCMatrixPrint(dropout1_forward);

        NNCIMatrixType dense2_forward = NNCDenseLayerForward(dropout1_forward, dense2);

//        NNCIMatrixType dense2_forward = NNCDenseLayerForward(relu1_forward, dense2);

//        NNCMatrixPrint(dense2_forward);
        NNCIMatrixType softmax1_forward = NNCActivationSoftMaxForward(dense2_forward);

        NNCIMatrixType ccel1_forward = NNCLossCCELForward(softmax1_forward, target);

        if(epoch % 1 == 0){
            nnc_mtype mean = NNCMatrixMean(ccel1_forward);
            nnc_vector argmax_prediction = NNCMatrixArgMax(softmax1_forward);
            nnc_vector argmax_target = NNCMatrixToVector(target, 1);
            nnc_mtype regularization_loss = NNCDenseLayerCalculateRegularizationLoss(dense1) + NNCDenseLayerCalculateRegularizationLoss(dense2);

            printf("epoch : %d ", epoch);
            printf(" lrate : %.9g ", optimizerAdam->current_learning_rate);
            printf("acc : %.9g", NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len));
            printf(" reg loss : %.6g", regularization_loss);
            printf(" mean loss : %.9g \n", mean);

            free(argmax_target);
            free(argmax_prediction);
        }

//        if(epoch == epoch_len) break; // Test epoch

        //---------BACKWARD PASS---------

//        NNCIMatrixType softmax_ccel_backward = NNCActivationSoftMaxLossCCELBackward(softmax1_forward, target);
//        NNCDenseLayerWithRegularizationBackward(softmax_ccel_backward, dense2);
//        NNCIMatrixType relu1_backward = NNCActivationReLUBackward(relu1_forward, dense2->dinputs);
//        NNCDenseLayerBackward(relu1_backward, dense1);

        NNCIMatrixType softmax_ccel_backward = NNCActivationSoftMaxLossCCELBackward(softmax1_forward, target);
        NNCDenseLayerBackward(softmax_ccel_backward, dense2);
        NNCDropoutLayerBackward(dense2->dinputs, dropout1);
        NNCIMatrixType relu1_backward = NNCActivationReLUBackward(relu1_forward, dropout1->dinputs);
        NNCDenseLayerWithRegularizationBackward(relu1_backward, dense1);

//        NNCOptimizerSGDPreUpdateParams(optimizerSgd);
//        NNCOptimizerSGDUpdateParams(optimizerSgd, dense1);
//        NNCOptimizerSGDUpdateParams(optimizerSgd, dense2);
//        NNCOptimizerSGDPostUpdateParams(optimizerSgd);

//        NNCOptimizerAdaGradPreUpdateParams(optimizerAdaGrad);
//        NNCOptimizerAdaGradUpdateParams(optimizerAdaGrad, dense1);
//        NNCOptimizerAdaGradUpdateParams(optimizerAdaGrad, dense2);
//        NNCOptimizerAdaGradPostUpdateParams(optimizerAdaGrad);

//        NNCOptimizerRMSPropPreUpdateParams(optimizerRmsProp);
//        NNCOptimizerRMSPropUpdateParams(optimizerRmsProp, dense1);
//        NNCOptimizerRMSPropUpdateParams(optimizerRmsProp, dense2);
//        NNCOptimizerRMSPropPostUpdateParams(optimizerRmsProp);

        NNCOptimizerAdamPreUpdateParams(optimizerAdam);
        NNCOptimizerAdamUpdateParams(optimizerAdam, dense1);
        NNCOptimizerAdamUpdateParams(optimizerAdam, dense2);
        NNCOptimizerAdamPostUpdateParams(optimizerAdam);


        NNCMatrixDeAlloc(dense1_forward);
        NNCMatrixDeAlloc(dense2_forward);
        NNCMatrixDeAlloc(relu1_forward);
        NNCMatrixDeAlloc(softmax1_forward);
        NNCMatrixDeAlloc(ccel1_forward);
        NNCMatrixDeAlloc(softmax_ccel_backward);
        NNCMatrixDeAlloc(relu1_backward);

//        NNCMatrixDeAlloc(softmax1_backward);
//        NNCMatrixDeAlloc(ccel1_backward);

    }

    printf("End\n");

    NNCMatrixDeAlloc(inputs_training);
    NNCMatrixDeAlloc(target_training);
    NNCMatrixDeAlloc(inputs_test);
    NNCMatrixDeAlloc(target_test);
    NNCDenseLayerDeAlloc(dense1);
    NNCDenseLayerDeAlloc(dense2);
    NNCOptimizerSGDDeAlloc(optimizerSgd);
    NNCOptimizerAdaGradDeAlloc(optimizerAdaGrad);
    NNCOptimizerRMSPropDeAlloc(optimizerRmsProp);
    NNCOptimizerAdamDeAlloc(optimizerAdam);

}

#endif //CMATRIX_DEVELOPMENT_H
