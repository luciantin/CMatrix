#ifndef CMATRIX_TESTS_H
#define CMATRIX_TESTS_H


#include <stdio.h>
#include "helpers/nnc_matrix.h"
#include "helpers/nnc_layer.h"
#include "helpers/nnc_activation_layer.h"
#include "helpers/nnc_loss_function.h"


NNCIMatrixType GetZeroOneMatrix_Test(){
    NNCIMatrixType zeroOneTest = NNCMatrixAlloc(5, 50);
    zeroOneTest->matrix[0][0] = 1.0;zeroOneTest->matrix[0][1] = 2.0;zeroOneTest->matrix[0][2] = 0.0;zeroOneTest->matrix[0][3] = 2.0;zeroOneTest->matrix[0][4] = 3.0;
    zeroOneTest->matrix[1][0] = 1.0;zeroOneTest->matrix[1][1] = 0.0;zeroOneTest->matrix[1][2] = 0.0;zeroOneTest->matrix[1][3] = 3.0;zeroOneTest->matrix[1][4] = 3.0;
    zeroOneTest->matrix[2][0] = 2.0;zeroOneTest->matrix[2][1] = 1.0;zeroOneTest->matrix[2][2] = 2.0;zeroOneTest->matrix[2][3] = 1.0;zeroOneTest->matrix[2][4] = 0.0;
    zeroOneTest->matrix[3][0] = 0.0;zeroOneTest->matrix[3][1] = 3.0;zeroOneTest->matrix[3][2] = 3.0;zeroOneTest->matrix[3][3] = 3.0;zeroOneTest->matrix[3][4] = 3.0;
    zeroOneTest->matrix[4][0] = 1.0;zeroOneTest->matrix[4][1] = 0.0;zeroOneTest->matrix[4][2] = 2.0;zeroOneTest->matrix[4][3] = 2.0;zeroOneTest->matrix[4][4] = 3.0;
    zeroOneTest->matrix[5][0] = 1.0;zeroOneTest->matrix[5][1] = 0.0;zeroOneTest->matrix[5][2] = 2.0;zeroOneTest->matrix[5][3] = 3.0;zeroOneTest->matrix[5][4] = 0.0;
    zeroOneTest->matrix[6][0] = 3.0;zeroOneTest->matrix[6][1] = 3.0;zeroOneTest->matrix[6][2] = 1.0;zeroOneTest->matrix[6][3] = 3.0;zeroOneTest->matrix[6][4] = 3.0;
    zeroOneTest->matrix[7][0] = 3.0;zeroOneTest->matrix[7][1] = 0.0;zeroOneTest->matrix[7][2] = 3.0;zeroOneTest->matrix[7][3] = 3.0;zeroOneTest->matrix[7][4] = 2.0;
    zeroOneTest->matrix[8][0] = 2.0;zeroOneTest->matrix[8][1] = 3.0;zeroOneTest->matrix[8][2] = 3.0;zeroOneTest->matrix[8][3] = 0.0;zeroOneTest->matrix[8][4] = 0.0;
    zeroOneTest->matrix[9][0] = 1.0;zeroOneTest->matrix[9][1] = 3.0;zeroOneTest->matrix[9][2] = 3.0;zeroOneTest->matrix[9][3] = 3.0;zeroOneTest->matrix[9][4] = 0.0;
    zeroOneTest->matrix[10][0] = 2.0;zeroOneTest->matrix[10][1] = 2.0;zeroOneTest->matrix[10][2] = 3.0;zeroOneTest->matrix[10][3] = 1.0;zeroOneTest->matrix[10][4] = 2.0;
    zeroOneTest->matrix[11][0] = 2.0;zeroOneTest->matrix[11][1] = 0.0;zeroOneTest->matrix[11][2] = 0.0;zeroOneTest->matrix[11][3] = 3.0;zeroOneTest->matrix[11][4] = 3.0;
    zeroOneTest->matrix[12][0] = 3.0;zeroOneTest->matrix[12][1] = 0.0;zeroOneTest->matrix[12][2] = 3.0;zeroOneTest->matrix[12][3] = 3.0;zeroOneTest->matrix[12][4] = 2.0;
    zeroOneTest->matrix[13][0] = 3.0;zeroOneTest->matrix[13][1] = 1.0;zeroOneTest->matrix[13][2] = 0.0;zeroOneTest->matrix[13][3] = 3.0;zeroOneTest->matrix[13][4] = 3.0;
    zeroOneTest->matrix[14][0] = 2.0;zeroOneTest->matrix[14][1] = 0.0;zeroOneTest->matrix[14][2] = 2.0;zeroOneTest->matrix[14][3] = 1.0;zeroOneTest->matrix[14][4] = 2.0;
    zeroOneTest->matrix[15][0] = 1.0;zeroOneTest->matrix[15][1] = 3.0;zeroOneTest->matrix[15][2] = 1.0;zeroOneTest->matrix[15][3] = 1.0;zeroOneTest->matrix[15][4] = 1.0;
    zeroOneTest->matrix[16][0] = 2.0;zeroOneTest->matrix[16][1] = 1.0;zeroOneTest->matrix[16][2] = 3.0;zeroOneTest->matrix[16][3] = 2.0;zeroOneTest->matrix[16][4] = 1.0;
    zeroOneTest->matrix[17][0] = 1.0;zeroOneTest->matrix[17][1] = 3.0;zeroOneTest->matrix[17][2] = 1.0;zeroOneTest->matrix[17][3] = 0.0;zeroOneTest->matrix[17][4] = 3.0;
    zeroOneTest->matrix[18][0] = 3.0;zeroOneTest->matrix[18][1] = 3.0;zeroOneTest->matrix[18][2] = 3.0;zeroOneTest->matrix[18][3] = 1.0;zeroOneTest->matrix[18][4] = 3.0;
    zeroOneTest->matrix[19][0] = 2.0;zeroOneTest->matrix[19][1] = 3.0;zeroOneTest->matrix[19][2] = 2.0;zeroOneTest->matrix[19][3] = 2.0;zeroOneTest->matrix[19][4] = 2.0;
    zeroOneTest->matrix[20][0] = 1.0;zeroOneTest->matrix[20][1] = 0.0;zeroOneTest->matrix[20][2] = 2.0;zeroOneTest->matrix[20][3] = 0.0;zeroOneTest->matrix[20][4] = 3.0;
    zeroOneTest->matrix[21][0] = 3.0;zeroOneTest->matrix[21][1] = 3.0;zeroOneTest->matrix[21][2] = 3.0;zeroOneTest->matrix[21][3] = 0.0;zeroOneTest->matrix[21][4] = 2.0;
    zeroOneTest->matrix[22][0] = 3.0;zeroOneTest->matrix[22][1] = 1.0;zeroOneTest->matrix[22][2] = 1.0;zeroOneTest->matrix[22][3] = 0.0;zeroOneTest->matrix[22][4] = 3.0;
    zeroOneTest->matrix[23][0] = 2.0;zeroOneTest->matrix[23][1] = 0.0;zeroOneTest->matrix[23][2] = 3.0;zeroOneTest->matrix[23][3] = 3.0;zeroOneTest->matrix[23][4] = 2.0;
    zeroOneTest->matrix[24][0] = 2.0;zeroOneTest->matrix[24][1] = 0.0;zeroOneTest->matrix[24][2] = 3.0;zeroOneTest->matrix[24][3] = 2.0;zeroOneTest->matrix[24][4] = 2.0;
    zeroOneTest->matrix[25][0] = 0.0;zeroOneTest->matrix[25][1] = 0.0;zeroOneTest->matrix[25][2] = 3.0;zeroOneTest->matrix[25][3] = 1.0;zeroOneTest->matrix[25][4] = 2.0;
    zeroOneTest->matrix[26][0] = 0.0;zeroOneTest->matrix[26][1] = 3.0;zeroOneTest->matrix[26][2] = 3.0;zeroOneTest->matrix[26][3] = 2.0;zeroOneTest->matrix[26][4] = 1.0;
    zeroOneTest->matrix[27][0] = 0.0;zeroOneTest->matrix[27][1] = 2.0;zeroOneTest->matrix[27][2] = 2.0;zeroOneTest->matrix[27][3] = 3.0;zeroOneTest->matrix[27][4] = 0.0;
    zeroOneTest->matrix[28][0] = 0.0;zeroOneTest->matrix[28][1] = 2.0;zeroOneTest->matrix[28][2] = 3.0;zeroOneTest->matrix[28][3] = 0.0;zeroOneTest->matrix[28][4] = 3.0;
    zeroOneTest->matrix[29][0] = 0.0;zeroOneTest->matrix[29][1] = 1.0;zeroOneTest->matrix[29][2] = 3.0;zeroOneTest->matrix[29][3] = 2.0;zeroOneTest->matrix[29][4] = 3.0;
    zeroOneTest->matrix[30][0] = 3.0;zeroOneTest->matrix[30][1] = 3.0;zeroOneTest->matrix[30][2] = 2.0;zeroOneTest->matrix[30][3] = 0.0;zeroOneTest->matrix[30][4] = 1.0;
    zeroOneTest->matrix[31][0] = 1.0;zeroOneTest->matrix[31][1] = 0.0;zeroOneTest->matrix[31][2] = 3.0;zeroOneTest->matrix[31][3] = 3.0;zeroOneTest->matrix[31][4] = 1.0;
    zeroOneTest->matrix[32][0] = 3.0;zeroOneTest->matrix[32][1] = 3.0;zeroOneTest->matrix[32][2] = 3.0;zeroOneTest->matrix[32][3] = 1.0;zeroOneTest->matrix[32][4] = 2.0;
    zeroOneTest->matrix[33][0] = 3.0;zeroOneTest->matrix[33][1] = 1.0;zeroOneTest->matrix[33][2] = 3.0;zeroOneTest->matrix[33][3] = 1.0;zeroOneTest->matrix[33][4] = 1.0;
    zeroOneTest->matrix[34][0] = 3.0;zeroOneTest->matrix[34][1] = 2.0;zeroOneTest->matrix[34][2] = 1.0;zeroOneTest->matrix[34][3] = 2.0;zeroOneTest->matrix[34][4] = 2.0;
    zeroOneTest->matrix[35][0] = 3.0;zeroOneTest->matrix[35][1] = 3.0;zeroOneTest->matrix[35][2] = 1.0;zeroOneTest->matrix[35][3] = 2.0;zeroOneTest->matrix[35][4] = 1.0;
    zeroOneTest->matrix[36][0] = 2.0;zeroOneTest->matrix[36][1] = 1.0;zeroOneTest->matrix[36][2] = 2.0;zeroOneTest->matrix[36][3] = 2.0;zeroOneTest->matrix[36][4] = 0.0;
    zeroOneTest->matrix[37][0] = 2.0;zeroOneTest->matrix[37][1] = 3.0;zeroOneTest->matrix[37][2] = 2.0;zeroOneTest->matrix[37][3] = 3.0;zeroOneTest->matrix[37][4] = 2.0;
    zeroOneTest->matrix[38][0] = 2.0;zeroOneTest->matrix[38][1] = 2.0;zeroOneTest->matrix[38][2] = 0.0;zeroOneTest->matrix[38][3] = 0.0;zeroOneTest->matrix[38][4] = 0.0;
    zeroOneTest->matrix[39][0] = 2.0;zeroOneTest->matrix[39][1] = 3.0;zeroOneTest->matrix[39][2] = 1.0;zeroOneTest->matrix[39][3] = 1.0;zeroOneTest->matrix[39][4] = 2.0;
    zeroOneTest->matrix[40][0] = 1.0;zeroOneTest->matrix[40][1] = 0.0;zeroOneTest->matrix[40][2] = 2.0;zeroOneTest->matrix[40][3] = 2.0;zeroOneTest->matrix[40][4] = 3.0;
    zeroOneTest->matrix[41][0] = 3.0;zeroOneTest->matrix[41][1] = 2.0;zeroOneTest->matrix[41][2] = 0.0;zeroOneTest->matrix[41][3] = 3.0;zeroOneTest->matrix[41][4] = 2.0;
    zeroOneTest->matrix[42][0] = 1.0;zeroOneTest->matrix[42][1] = 3.0;zeroOneTest->matrix[42][2] = 0.0;zeroOneTest->matrix[42][3] = 1.0;zeroOneTest->matrix[42][4] = 1.0;
    zeroOneTest->matrix[43][0] = 1.0;zeroOneTest->matrix[43][1] = 2.0;zeroOneTest->matrix[43][2] = 2.0;zeroOneTest->matrix[43][3] = 2.0;zeroOneTest->matrix[43][4] = 2.0;
    zeroOneTest->matrix[44][0] = 2.0;zeroOneTest->matrix[44][1] = 3.0;zeroOneTest->matrix[44][2] = 1.0;zeroOneTest->matrix[44][3] = 2.0;zeroOneTest->matrix[44][4] = 3.0;
    zeroOneTest->matrix[45][0] = 2.0;zeroOneTest->matrix[45][1] = 2.0;zeroOneTest->matrix[45][2] = 3.0;zeroOneTest->matrix[45][3] = 1.0;zeroOneTest->matrix[45][4] = 2.0;
    zeroOneTest->matrix[46][0] = 1.0;zeroOneTest->matrix[46][1] = 2.0;zeroOneTest->matrix[46][2] = 2.0;zeroOneTest->matrix[46][3] = 0.0;zeroOneTest->matrix[46][4] = 3.0;
    zeroOneTest->matrix[47][0] = 0.0;zeroOneTest->matrix[47][1] = 1.0;zeroOneTest->matrix[47][2] = 3.0;zeroOneTest->matrix[47][3] = 0.0;zeroOneTest->matrix[47][4] = 0.0;
    zeroOneTest->matrix[48][0] = 1.0;zeroOneTest->matrix[48][1] = 1.0;zeroOneTest->matrix[48][2] = 0.0;zeroOneTest->matrix[48][3] = 1.0;zeroOneTest->matrix[48][4] = 0.0;
    zeroOneTest->matrix[49][0] = 2.0;zeroOneTest->matrix[49][1] = 1.0;zeroOneTest->matrix[49][2] = 3.0;zeroOneTest->matrix[49][3] = 2.0;zeroOneTest->matrix[49][4] = 1.0;

    return zeroOneTest;
}

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
//    NNCIMatrixType normalizedReLUBackward = NNCActivationReLUBackward(dvalues);
//    NNCMatrixPrint(normalizedReLUBackward);


    puts("--------------");
    puts("softmax backward");
    NNCIMatrixType softmaxBackward = NNCActivationSoftMaxBackward(dvalues, normalized1);
    NNCMatrixPrint(softmaxBackward);

}


#endif //CMATRIX_TESTS_H
