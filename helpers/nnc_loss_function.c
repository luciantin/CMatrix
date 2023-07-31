#include <malloc.h>
#include "nnc_loss_function.h"
#include <math.h>
#include <stdlib.h>

NNCIMatrixType NNCLossCCELForward(NNCIMatrixType prediction, NNCIMatrixType target){
    NNCIMatrixType output = NNCMatrixAlloc(1, prediction->y);
    NNCIMatrixType input_ = NNCMatrixClip(prediction, 1e-7, 1 - 1e-7);

    // one-hot encoded labels
    if(prediction->x == target->x && target->x == 1){
        for(int _y = 0; _y < input_->y; _y ++) {
            double loss = 0;
            for (int _x = 0; _x < input_->x; _x++){
                loss += (nnc_mtype)input_->matrix[_y][_x] * (nnc_mtype)target->matrix[_y][_x];
            }
            output->matrix[_y][0] = -log(loss);
        }
    }
    // categorical labels
    else if(target->x == 1){
        for(int _y = 0; _y < input_->y; _y ++){
            int index = (int)target->matrix[_y][0];
            nnc_mtype pred = (nnc_mtype)input_->matrix[_y][index];
            output->matrix[_y][0] = -log(pred);
        }

    }

    NNCMatrixDeAlloc(input_);
    return output;
}

NNCIMatrixType NNCLossCCELBackward(NNCIMatrixType dvalues, NNCIMatrixType target){
    NNCIMatrixType output = NNCMatrixAllocBaseValue(dvalues->x, dvalues->y, 0);

    if(target->x > 1){
        for(int _y = 0; _y < output->y; _y ++) for(int _x = 0; _x < output->x; _x++)
                //                                          calculate the gradient                and normalize it
                output->matrix[_y][_x] = ( - target->matrix[_y][_x] / dvalues->matrix[_y][_x] ) / dvalues->y;
    }
    else if(target->x == 1){
        for(int _y = 0; _y < output->y; _y ++) output->matrix[_y][(int)target->matrix[_y][0]] = ( - 1 / dvalues->matrix[_y][(int)target->matrix[_y][0]] ) / dvalues->y;
    }

    return output;
}