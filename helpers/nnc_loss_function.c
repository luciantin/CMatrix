#include <malloc.h>
#include "nnc_loss_function.h"
#include <math.h>
#include <stdlib.h>

NNCIMatrixType NNCLossCCELForward(NNCIMatrixType input, NNCIMatrixType target){
    NNCIMatrixType output = NNCMatrixAlloc(input->y, 1);
    for(int _y = 0; _y < input->y; _y ++) {
        double loss = 0;
        for (int _x = 0; _x < input->x; _x++){
            loss += (double)input->matrix[_y][_x] * (double)target->matrix[_y][_x];
        }
        output->matrix[0][_y] = -log(loss);
    }
    return output;
}

NNCIMatrixType NNCLossCCELBackward(NNCIMatrixType dvalues, NNCIMatrixType target){
    NNCIMatrixType output = NNCMatrixAlloc(dvalues->x, dvalues->y);
    for(int _y = 0; _y < output->y; _y ++) for(int _x = 0; _x < output->x; _x++)
        //                                          calculate the gradient                and normalize it
        output->matrix[_y][_x] = ( - target->matrix[_y][_x] / dvalues->matrix[_y][_x] ) / dvalues->y;
    return output;
}