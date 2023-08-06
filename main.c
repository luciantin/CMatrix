#include <stdio.h>
#include <stdlib.h>
#include "helpers/nnc_config.h"
#include "helpers/nnc_matrix.h"
#include "helpers/nnc_dense_layer.h"
#include "helpers/nnc_activation_layer.h"
#include "helpers/nnc_loss_function.h"
#include "helpers/nnc_vector.h"
#include "tests.h"
#include "helpers/nnc_optimizer.h"
#include "AutoGenTest.h"


int main() {
    printf("Start\n");
    puts("--------------");

    NNCIMatrixType inputs = GetAutoGenTestMatrix();
    NNCIMatrixType target = GetAutoGenTruthMatrix();

    int sample_len = inputs->y;
    int input_len = inputs->x;
    int output_len = 3;
    nnc_mtype learning_rate = 1;
    nnc_mtype decay = 1e-6;
    int epoch_len = 1000;
    nnc_mtype momentum = 0;

    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(input_len, 124);
    NNCIDenseLayerType dense2 = NNCDenseLayerAlloc(124, output_len);
    NNCIOptimizerSGDType optimizerSgd = NNCOptimizerSGDAlloc(learning_rate, decay, momentum);
    NNCIOptimizerAdaGradType optimizerAdaGrad = NNCOptimizerAdaGradAlloc(learning_rate, decay);

    //---------FORWARD PASS---------
    for(int epoch = 0; epoch < epoch_len; epoch ++){
        NNCIMatrixType dense1_forward = NNCDenseLayerForward(inputs, dense1);
//        NNCMatrixPrint(dense1_forward);
        NNCIMatrixType relu1_forward = NNCActivationReLUForward(dense1_forward);
//        NNCMatrixPrint(relu1_forward);
        NNCIMatrixType dense2_forward = NNCDenseLayerForward(relu1_forward, dense2);
//        NNCMatrixPrint(dense2_forward);

        NNCIMatrixType softmax1_forward = NNCActivationSoftMaxForward(dense2_forward);
//        NNCMatrixPrint(softmax1_forward);

        NNCIMatrixType ccel1_forward = NNCLossCCELForward(softmax1_forward, target);
//        NNCMatrixPrint(ccel1_forward);

        if(epoch % 1 == 0){
            nnc_mtype mean = NNCMatrixMean(ccel1_forward);
            nnc_vector argmax_prediction = NNCMatrixArgMax(softmax1_forward);
            nnc_vector argmax_target = NNCMatrixToVector(target, 1);

            printf("epoch : %d ", epoch);
            printf(" lrate : %f ", optimizerAdaGrad->current_learning_rate);
            printf("acc : %f", NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len));
            printf(" mean loss : %f \n", mean);
//            puts("--------------");

            free(argmax_target);
            free(argmax_prediction);
        }

        //---------BACKWARD PASS---------

//        NNCIMatrixType ccel1_backward = NNCLossCCELBackward(softmax1_forward, target);
////        NNCMatrixPrint(ccel1_backward);
//
//        NNCIMatrixType softmax1_backward = NNCActivationSoftMaxBackward(ccel1_backward, softmax1_forward);
////        NNCMatrixPrint(softmax1_backward);

        NNCIMatrixType softmax_ccel_backward = NNCActivationSoftMaxLossCCELBackward(softmax1_forward, target);
//        NNCMatrixPrint(softmax_ccel_backward);

        NNCDenseLayerBackward(softmax_ccel_backward, dense2);
//        NNCMatrixPrint(dense2->dinputs);

        NNCIMatrixType relu1_backward = NNCActivationReLUBackward(relu1_forward, dense2->dinputs);
//        NNCMatrixPrint(relu1_backward);

        NNCDenseLayerBackward(relu1_backward, dense1);
//        NNCMatrixPrint(dense1->dinputs);
//        puts("--------------");

//        NNCOptimizerSGDPreUpdateParams(optimizerSgd);
//        NNCOptimizerSGDUpdateParams(optimizerSgd, dense1);
//        NNCOptimizerSGDUpdateParams(optimizerSgd, dense2);
//        NNCOptimizerSGDPostUpdateParams(optimizerSgd);

        NNCOptimizerAdaGradPreUpdateParams(optimizerAdaGrad);
        NNCOptimizerAdaGradUpdateParams(optimizerAdaGrad, dense1);
        NNCOptimizerAdaGradUpdateParams(optimizerAdaGrad, dense2);
        NNCOptimizerAdaGradPostUpdateParams(optimizerAdaGrad);

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

    NNCMatrixDeAlloc(inputs);
    NNCMatrixDeAlloc(target);
    NNCDenseLayerDeAlloc(dense1);
    NNCDenseLayerDeAlloc(dense2);
    NNCOptimizerSGDDeAlloc(optimizerSgd);
    NNCOptimizerAdaGradDeAlloc(optimizerAdaGrad);

    return 0;
}
