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

    NNCIMatrixType inputs = NNCMatrixAlloc(1, 10);
    inputs->matrix[0][0] = 1;
    inputs->matrix[1][0] = 2;
    inputs->matrix[2][0] = 1;
    inputs->matrix[3][0] = 2;
    inputs->matrix[4][0] = 1;
    inputs->matrix[5][0] = 2;
    inputs->matrix[6][0] = 1;
    inputs->matrix[7][0] = 2;
    inputs->matrix[8][0] = 1;
    inputs->matrix[9][0] = 2;

    NNCIMatrixType target = NNCMatrixAlloc(1, 10);
    target->matrix[0][0] = 1;
    target->matrix[1][0] = 2;
    target->matrix[2][0] = 1;
    target->matrix[3][0] = 2;
    target->matrix[4][0] = 1;
    target->matrix[5][0] = 2;
    target->matrix[6][0] = 1;
    target->matrix[7][0] = 2;
    target->matrix[8][0] = 1;
    target->matrix[9][0] = 2;

//    NNCMatrixPrint(inputs);

    puts("--------------");

    srand(231423);
    //    NNCIDenseLayerType dense123 = NNCDenseLayerAlloc(1, 2);
    NNCIDenseLayerType dense1 = NNCDenseLayerAlloc(1, 2);
    NNCIMatrixType output1 = NNCDenseLayerForward(inputs, dense1);
    NNCIMatrixType normalized1 = NNCActivationSoftMaxForward(output1);
    nnc_vector loss1 = NNCLossCCEL(normalized1, target);

    NNCMatrixPrint(dense1->weights);
    puts("--------------");

    NNCMatrixPrint(output1);
    puts("--------------");

    NNCMatrixPrint(normalized1);
    NNCVectorPrint(loss1, normalized1->y);


    puts("--------------");

//    printf("Mean : %f", NNCVectorMean(loss1, normalized1->y));
    NNCIMatrixType matrixTest = NNCMatrixAlloc(3,3);
    matrixTest->matrix[0][0] = 0.7;  matrixTest->matrix[0][1] = 0.2; matrixTest->matrix[0][2] = 0.1;
    matrixTest->matrix[1][0] = 0.5;  matrixTest->matrix[1][1] = 0.1; matrixTest->matrix[1][2] = 0.4;
    matrixTest->matrix[2][0] = 0.02; matrixTest->matrix[2][1] = 0.9; matrixTest->matrix[2][2] = 0.08;

    NNCVectorPrint(NNCMatrixArgMax(matrixTest), 3);

    printf("End\n");
    return 0;
}
