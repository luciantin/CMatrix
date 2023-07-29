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


int main() {
    printf("Start\n");
    puts("--------------");

    // 4 inputs
    NNCIMatrixType inputs = GetSumInputMatrix_Test();
    // 3 outputs
    NNCIMatrixType target = GetSumTargetMatrix_Test();
    int sample_len = 10;

    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(4, 64);
    NNCIDenseLayerType dense2 = NNCDenseLayerAlloc(64, 3);

    //---------FORWARD PASS---------
    for(int epoch = 0; epoch < 50; epoch ++){
        NNCIMatrixType dense1_output = NNCDenseLayerForward(inputs, dense1);
        NNCIMatrixType relu1_output = NNCActivationReLUForward(dense1_output);

        NNCIMatrixType dense2_output = NNCDenseLayerForward(relu1_output, dense2);
        NNCIMatrixType softmax1_output = NNCActivationSoftMaxForward(dense2_output);
        NNCIMatrixType ccel1_output = NNCLossCCELForward(softmax1_output, target);

        nnc_mtype mean = NNCMatrixMean(ccel1_output);
        nnc_vector argmax_prediction = NNCMatrixArgMax(softmax1_output);
        nnc_vector argmax_target = NNCMatrixArgMax(target);

        printf("loss :");
        NNCMatrixPrint(ccel1_output);
        printf("acc : %f\n", NNCVectorAccuracy(argmax_target, argmax_prediction, sample_len));
        printf("mean : %f \n", mean);
        printf("target :     ");
        NNCVectorPrint(argmax_target, sample_len);
        printf("prediction : ");
        NNCVectorPrint(argmax_prediction, sample_len);
        puts("--------------");

        //---------BACKWARD PASS---------

        NNCIMatrixType ccel1_backward = NNCLossCCELBackward(softmax1_output, target);
        NNCIMatrixType softmax1_backward = NNCActivationSoftMaxBackward(ccel1_backward, softmax1_output);
        NNCDenseLayerBackward(softmax1_backward, dense2);
        NNCIMatrixType relu1_backward = NNCActivationReLUBackward(dense2->dinputs);
        NNCDenseLayerBackward(relu1_backward, dense1);

        NNCIOptimizerSGDType optimizerSgd = NNCOptimizerSGDAlloc(0.0001);
        NNCOptimizerSGDUpdateParams(optimizerSgd, dense1);
        NNCOptimizerSGDUpdateParams(optimizerSgd, dense2);

        NNCMatrixDeAlloc(dense1_output);
        NNCMatrixDeAlloc(dense2_output);
        NNCMatrixDeAlloc(relu1_output);
        NNCMatrixDeAlloc(softmax1_output);
        NNCMatrixDeAlloc(ccel1_output);
    }

    puts("--------------");
    printf("End\n");

    NNCMatrixDeAlloc(inputs);
    NNCMatrixDeAlloc(target);
    NNCDenseLayerDeAlloc(dense1);
    NNCDenseLayerDeAlloc(dense2);


    return 0;
}
