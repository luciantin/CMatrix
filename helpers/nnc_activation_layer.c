#include "nnc_activation_layer.h"

NNCIMatrixType NNCActivationReLUForward(NNCIMatrixType input) {
    NNCIMatrixType output  = NNCMatrixAlloc(input->x, input->y);
    for(int _y = 0; _y < input->y; _y ++)
        for(int _x = 0; _x < input->x; _x ++)
            output->matrix[_y][_x] = input->matrix[_y][_x] > 0 ? input->matrix[_y][_x] : 0;
    return output;
}

// todo za kategorizaciju, nije bas savrsena
NNCIMatrixType NNCActivationSoftMaxForward(NNCIMatrixType input) {
    NNCIMatrixType output  = NNCMatrixAllocBaseValue(input->x, input->y, 0);
    for(int _y = 0; _y < input->y; _y ++){
        double row_sum = 0;
        for(int _x = 0; _x < input->x; _x ++) row_sum += input->matrix[_y][_x] * input->matrix[_y][_x];
        if(row_sum == 0) continue;
        for(int _x = 0; _x < input->x; _x ++){
            double row_exp = input->matrix[_y][_x] * input->matrix[_y][_x];
            output->matrix[_y][_x] = (row_exp / row_sum); //((unsigned char)(1 << input->matrix[_y][_x])) / row_sum;
        }
    }
    return output;
}
