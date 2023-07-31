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
    double learning_rate = 1;
    nnc_mtype decay = 0;
    int epoch_len = 1001;

    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(input_len, 6);
    NNCIDenseLayerType dense2 = NNCDenseLayerAlloc(6, output_len);
    NNCIOptimizerSGDType optimizerSgd = NNCOptimizerSGDAlloc(learning_rate, decay);

    //---------FORWARD PASS---------
    for(int epoch = 0; epoch < epoch_len; epoch ++){
        NNCIMatrixType dense1_forward = NNCDenseLayerForward(inputs, dense1);
        NNCIMatrixType relu1_forward = NNCActivationReLUForward(dense1_forward);
        NNCIMatrixType dense2_forward = NNCDenseLayerForward(relu1_forward, dense2);
        NNCIMatrixType softmax1_forward = NNCActivationSoftMaxForward(dense2_forward);
        NNCIMatrixType ccel1_forward = NNCLossCCELForward(softmax1_forward, target);

        if(epoch % 1 == 0){
            nnc_mtype mean = NNCMatrixMean(ccel1_forward);
            nnc_vector argmax_prediction = NNCMatrixArgMax(softmax1_forward);
            nnc_vector argmax_target = NNCMatrixToVector(target, 1);

            printf("epoch : %d ", epoch);
            printf(" lrate : %f ", optimizerSgd->current_learning_rate);
            printf("acc : %f", NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len));
            printf(" mean loss : %f \n", mean);
            puts("--------------");

            free(argmax_target);
            free(argmax_prediction);
        }

        //---------BACKWARD PASS---------

        NNCIMatrixType ccel1_backward = NNCLossCCELBackward(softmax1_forward, target);
        NNCIMatrixType softmax1_backward = NNCActivationSoftMaxBackward(ccel1_backward, softmax1_forward);

        NNCDenseLayerBackward(softmax1_backward, dense2);
//        NNCMatrixPrint(dense2->dinputs);

        NNCIMatrixType relu1_backward = NNCActivationReLUBackward(relu1_forward, dense2->dinputs);
//        NNCMatrixPrint(relu1_backward);

        NNCDenseLayerBackward(relu1_backward, dense1);
//        NNCMatrixPrint(dense1->dinputs);
//        puts("--------------");

        NNCOptimizerSGDPreUpdateParams(optimizerSgd);
        NNCOptimizerSGDUpdateParams(optimizerSgd, dense1);
        NNCOptimizerSGDUpdateParams(optimizerSgd, dense2);
        NNCOptimizerSGDPostUpdateParams(optimizerSgd);

        NNCMatrixDeAlloc(dense1_forward);
        NNCMatrixDeAlloc(dense2_forward);
        NNCMatrixDeAlloc(relu1_forward);
        NNCMatrixDeAlloc(softmax1_forward);
        NNCMatrixDeAlloc(ccel1_forward);

        NNCMatrixDeAlloc(relu1_backward);
        NNCMatrixDeAlloc(softmax1_backward);
        NNCMatrixDeAlloc(ccel1_backward);
        
    }

    printf("End\n");

    NNCMatrixDeAlloc(inputs);
    NNCMatrixDeAlloc(target);
    NNCDenseLayerDeAlloc(dense1);
    NNCDenseLayerDeAlloc(dense2);
    NNCOptimizerSGDDeAlloc(optimizerSgd);

    return 0;
}
