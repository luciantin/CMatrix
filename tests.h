#ifndef CMATRIX_TESTS_H
#define CMATRIX_TESTS_H


#include <stdio.h>
#include "helpers/nnc_matrix.h"
#include "helpers/nnc_dense_layer.h"
#include "helpers/nnc_activation_layer.h"
#include "helpers/nnc_loss_function.h"

// dali je suma reda negativna, 0 ili pozitivna
NNCIMatrixType GetSumInputMatrix_Test(){
    NNCIMatrixType inputs = NNCMatrixAlloc(4, 10);
    inputs->matrix[0][0] = 1; inputs->matrix[0][1] = 2; inputs->matrix[0][2] = 1; inputs->matrix[0][3] = -1;
    inputs->matrix[1][0] = 1; inputs->matrix[1][1] = 5; inputs->matrix[1][2] = 1; inputs->matrix[1][3] = -10;
    inputs->matrix[2][0] = 4; inputs->matrix[2][1] = 1; inputs->matrix[2][2] = 4; inputs->matrix[2][3] = 1;
    inputs->matrix[3][0] = 1; inputs->matrix[3][1] = -6; inputs->matrix[3][2] = 1; inputs->matrix[3][3] = 1;
    inputs->matrix[4][0] = 1; inputs->matrix[4][1] = 1; inputs->matrix[4][2] = -9; inputs->matrix[4][3] = 5;

    inputs->matrix[5][0] = 4; inputs->matrix[5][1] = 5; inputs->matrix[5][2] = 0; inputs->matrix[5][3] = -7;
    inputs->matrix[6][0] = 1; inputs->matrix[6][1] = 1; inputs->matrix[6][2] = 5; inputs->matrix[6][3] = 4;
    inputs->matrix[7][0] = 1; inputs->matrix[7][1] = 1; inputs->matrix[7][2] = 10; inputs->matrix[7][3] = 1;
    inputs->matrix[8][0] = 1; inputs->matrix[8][1] = 1; inputs->matrix[8][2] = 4; inputs->matrix[8][3] = 1;
    inputs->matrix[9][0] = 1; inputs->matrix[9][1] = 1; inputs->matrix[9][2] = 8; inputs->matrix[9][3] = 1;
    return inputs;
}
NNCIMatrixType GetSumTargetMatrix_Test(){
    NNCIMatrixType target = NNCMatrixAlloc(3, 10);
    target->matrix[0][0] = 0; target->matrix[0][1] = 0; target->matrix[0][2] = 1;
    target->matrix[1][0] = 1; target->matrix[1][1] = 0; target->matrix[1][2] = 0;
    target->matrix[2][0] = 0; target->matrix[2][1] = 0; target->matrix[2][2] = 1;
    target->matrix[3][0] = 1; target->matrix[3][1] = 0; target->matrix[3][2] = 0;
    target->matrix[4][0] = 0; target->matrix[4][1] = 1; target->matrix[4][2] = 0;

    target->matrix[5][0] = 1; target->matrix[5][1] = 0; target->matrix[5][2] = 0;
    target->matrix[6][0] = 0; target->matrix[6][1] = 1; target->matrix[6][2] = 0;
    target->matrix[7][0] = 0; target->matrix[7][1] = 1; target->matrix[7][2] = 0;
    target->matrix[8][0] = 0; target->matrix[8][1] = 1; target->matrix[8][2] = 0;
    target->matrix[9][0] = 0; target->matrix[9][1] = 1; target->matrix[9][2] = 0;
    return target;
}


void TestAll(){
    NNCIMatrixType inputs = GetSumInputMatrix_Test();

    puts("inputs");
    NNCMatrixPrint(inputs);
    puts("--------------");

    NNCIMatrixType target = GetSumTargetMatrix_Test();

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

}


#endif //CMATRIX_TESTS_H
