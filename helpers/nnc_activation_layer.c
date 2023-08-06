#include <stdio.h>
#include "nnc_activation_layer.h"
#include <math.h>

NNCIMatrixType NNCActivationReLUForward(NNCIMatrixType input) {
    NNCIMatrixType output  = NNCMatrixAlloc(input->x, input->y);
    for(int _y = 0; _y < input->y; _y ++)
        for(int _x = 0; _x < input->x; _x ++)
            output->matrix[_y][_x] = input->matrix[_y][_x] > 0 ? input->matrix[_y][_x] : 0;
    return output;
}

NNCIMatrixType NNCActivationReLUBackward(NNCIMatrixType input, NNCIMatrixType dvalues) {
    NNCIMatrixType output  = NNCMatrixAlloc(input->x, input->y);
    for(int _y = 0; _y < input->y; _y ++)
        for(int _x = 0; _x < input->x; _x ++)
            output->matrix[_y][_x] = input->matrix[_y][_x] <= 0 ? 0 : dvalues->matrix[_y][_x];
    return output;
}

// todo za kategorizaciju, nije bas savrsena
NNCIMatrixType NNCActivationSoftMaxForward(NNCIMatrixType input) {
    NNCIMatrixType output  = NNCMatrixAllocBaseValue(input->x, input->y, 0);
    for(int _y = 0; _y < input->y; _y ++){
        nnc_mtype row_max = input->matrix[_y][0];
        nnc_mtype row_sum = 0;
        for(int _x = 0; _x < input->x; _x ++) row_max = input->matrix[_y][_x] > row_max ? input->matrix[_y][_x] : row_max;
        for(int _x = 0; _x < input->x; _x ++) {
            output->matrix[_y][_x] = expf(input->matrix[_y][_x] - row_max);
            row_sum += output->matrix[_y][_x];
        }
        for(int _x = 0; _x < input->x; _x ++)
            output->matrix[_y][_x] = output->matrix[_y][_x] / row_sum;
    }

    return output;
}

NNCIMatrixType NNCActivationSoftMaxBackward(NNCIMatrixType dvalues, NNCIMatrixType softmax_output){
    NNCIMatrixType output = NNCMatrixAlloc(softmax_output->x, softmax_output->y);
    for(int _y = 0; _y < softmax_output->y; _y ++){

        NNCIMatrixType _row_vector = NNCMatrixAlloc(softmax_output->x, 1);
        for(int _x = 0; _x < softmax_output->x; _x ++) _row_vector->matrix[0][_x] = softmax_output->matrix[_y][_x];

        NNCIMatrixType _diagflat = NNCMatrixAllocBaseValue(softmax_output->x, softmax_output->x, 0);
        for(int _x = 0; _x < softmax_output->x; _x ++) _diagflat->matrix[_x][_x] = softmax_output->matrix[_y][_x];

        NNCIMatrixType _transposed = NNCMatrixTranspose(_row_vector);
        NNCIMatrixType _product = NNCMatrixProduct(_transposed, _row_vector);
        NNCIMatrixType _jacobian_matrix = NNCMatrixSub(_diagflat, _product);

        NNCIMatrixType _row_dvector = NNCMatrixAlloc(dvalues->x, 1);
        for(int _x = 0; _x < dvalues->x; _x ++) _row_dvector->matrix[0][_x] = dvalues->matrix[_y][_x];

        NNCIMatrixType _sample_wise_gradient = NNCMatrixProduct(_row_dvector, _jacobian_matrix);
        for(int _x = 0; _x < output->x; _x ++) output->matrix[_y][_x] = _sample_wise_gradient->matrix[0][_x];

        NNCMatrixDeAlloc(_row_vector);
        NNCMatrixDeAlloc(_diagflat);
        NNCMatrixDeAlloc(_transposed);
        NNCMatrixDeAlloc(_product);
        NNCMatrixDeAlloc(_jacobian_matrix);
        NNCMatrixDeAlloc(_row_dvector);
        NNCMatrixDeAlloc(_sample_wise_gradient);
    }

    return output;
}