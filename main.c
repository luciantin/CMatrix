#include <stdio.h>
#include <stdlib.h>
#include "helpers/nnc_config.h"
#include "helpers/nnc_matrix.h"
#include "helpers/nnc_dense_layer.h"
#include "helpers/nnc_activation_layer.h"
#include "helpers/nnc_loss_function.h"
#include "helpers/nnc_vector.h"


int main() {
    printf("Start\n");
    puts("--------------");

    NNCIMatrixType inputs = NNCMatrixAlloc(4, 5);
    inputs->matrix[0][0] = 1; inputs->matrix[0][1] = 2; inputs->matrix[0][2] = 1; inputs->matrix[0][3] = -1;
    inputs->matrix[1][0] = 1; inputs->matrix[1][1] = 5; inputs->matrix[1][2] = 1; inputs->matrix[1][3] = -10;
    inputs->matrix[2][0] = 4; inputs->matrix[2][1] = 1; inputs->matrix[2][2] = 4; inputs->matrix[2][3] = 1;
    inputs->matrix[3][0] = 1; inputs->matrix[3][1] = -13; inputs->matrix[3][2] = 1; inputs->matrix[3][3] = 1;
    inputs->matrix[4][0] = 1; inputs->matrix[4][1] = 1; inputs->matrix[4][2] = -3; inputs->matrix[4][3] = 1;

    puts("inputs");
    NNCMatrixPrint(inputs);
    puts("--------------");

    // dali je suma reda negativna, 0 ili pozitivna
    NNCIMatrixType target = NNCMatrixAlloc(3, 5);
    target->matrix[0][0] = 0; target->matrix[0][1] = 0; target->matrix[0][2] = 1;
    target->matrix[1][0] = 1; target->matrix[1][1] = 0; target->matrix[1][2] = 0;
    target->matrix[2][0] = 0; target->matrix[2][1] = 0; target->matrix[2][2] = 1;
    target->matrix[3][0] = 1; target->matrix[3][1] = 0; target->matrix[3][2] = 0;
    target->matrix[4][0] = 0; target->matrix[4][1] = 1; target->matrix[4][2] = 0;

    puts("target");
    NNCMatrixPrint(target);
    puts("--------------");

    puts("weights");
    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(4, 3);
    NNCMatrixPrint(dense1->weights);
    puts("--------------");

    puts("biases");
    NNCMatrixPrint(dense1->biases);
    puts("--------------");

    puts("output");
    NNCIMatrixType output1 = NNCDenseLayerForward(inputs, dense1);
    NNCMatrixPrint(output1);

    puts("--------------");
    puts("ReLU forward");
    NNCIMatrixType normalizedReLU1 = NNCActivationReLUForward(output1);
    NNCMatrixPrint(normalizedReLU1);
    puts("--------------");

    puts("SoftMax forward");
    NNCIMatrixType normalized1 = NNCActivationSoftMaxForward(output1);
    NNCMatrixPrint(normalized1);
    puts("--------------");

    puts("NNCLossCCEL forward");
    NNCIMatrixType lossCCEL = NNCLossCCELForward(normalized1, target);
    NNCMatrixPrint(lossCCEL);
    puts("--------------");

    puts("dvalues");
    NNCIMatrixType dvalues = NNCMatrixAlloc(3, 5);
    dvalues->matrix[0][0] = 3; dvalues->matrix[0][1] = 4; dvalues->matrix[0][2] = -5;
    dvalues->matrix[1][0] = -4; dvalues->matrix[1][1] = 5; dvalues->matrix[1][2] = 6;
    dvalues->matrix[2][0] = 5; dvalues->matrix[2][1] = -6; dvalues->matrix[2][2] = 7;
    dvalues->matrix[3][0] = -6; dvalues->matrix[3][1] = 7; dvalues->matrix[3][2] = 8;
    dvalues->matrix[4][0] = 7; dvalues->matrix[4][1] = 8; dvalues->matrix[4][2] = -9;

    NNCMatrixPrint(dvalues);

    puts("--------------");
    NNCDenseLayerBackward(dvalues, dense1);
    puts("dinputs");
    NNCMatrixPrint(dense1->dinputs);

    puts("--------------");
    puts("dweights");
    NNCMatrixPrint(dense1->dweights);

    puts("--------------");
    puts("dbiases");
    NNCMatrixPrint(dense1->dbiases);

    puts("--------------");
    puts("ReLU backward");
    NNCIMatrixType normalizedReLUBackward = NNCActivationReLUBackward(dvalues);
    NNCMatrixPrint(normalizedReLUBackward);


    puts("--------------");
    puts("softmax backward");
    NNCIMatrixType softmaxBackward = NNCActivationSoftMaxBackward(dvalues, normalized1);
    NNCMatrixPrint(softmaxBackward);




//    NNCMatrixPrint(inputs);
//    puts("--------------");
//
//    NNCMatrixPrint(NNCMatrixSumSingle(inputs, 0));
//    puts("--------------");
//    srand(231423);
//    //    NNCIDenseLayerType dense123 = NNCDenseLayerAlloc(1, 2);
//    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(4, 2);
//    NNCIMatrixType output1 = NNCDenseLayerForward(inputs, dense1);
//    NNCIMatrixType normalized1 = NNCActivationSoftMaxForward(output1);
//    nnc_vector loss1 = NNCLossCCEL(normalized1, target);

//    NNCMatrixPrint(dense1->weights);
//    puts("--------------");
//    NNCMatrixPrint(dense1->biases);


//    NNCMatrixPrint(NNCMatrixSum(dense1->weights, dense1->biases));
//    puts("--------------");

//    NNCMatrixPrint(output1);
//    puts("--------------");
//
//    NNCMatrixPrint(normalized1);
//    NNCVectorPrint(loss1, normalized1->y);
//
//
//    puts("--------------");
//
////    printf("Mean : %f", NNCVectorMean(loss1, normalized1->y));
//    NNCIMatrixType matrixTest = NNCMatrixAlloc(3,3);
//    matrixTest->matrix[0][0] = 0.7;  matrixTest->matrix[0][1] = 0.2; matrixTest->matrix[0][2] = 0.1;
//    matrixTest->matrix[1][0] = 0.5;  matrixTest->matrix[1][1] = 0.1; matrixTest->matrix[1][2] = 0.4;
//    matrixTest->matrix[2][0] = 0.02; matrixTest->matrix[2][1] = 0.9; matrixTest->matrix[2][2] = 0.08;
//
//    NNCVectorPrint(NNCMatrixArgMax(matrixTest), 3);
    puts("--------------");
    printf("End\n");
    return 0;
}
