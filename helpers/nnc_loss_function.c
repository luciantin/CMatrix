#include <malloc.h>
#include "nnc_loss_function.h"
#include <math.h>
#include <stdlib.h>

nnc_vector NNCLossCCEL(NNCIMatrixType input, NNCIMatrixType target){
    nnc_vector output = malloc(sizeof(nnc_mtype)*input->y);
    for(int _y = 0; _y < input->y; _y ++) {
        double loss = 0;
        for (int _x = 0; _x < input->x; _x++){
            loss += log((double)input->matrix[_y][_x]) * (double)target->matrix[_y][_x];
        }
        output[_y] = fabs(loss);
    }
    return output;
}
