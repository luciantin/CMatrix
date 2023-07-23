#ifndef CMATRIX_NNC_ACTIVATION_LAYER_H
#define CMATRIX_NNC_ACTIVATION_LAYER_H

#include "nnc_matrix.h"

NNCIMatrixType NNCActivationReLUForward(NNCIMatrixType input);
NNCIMatrixType NNCActivationSoftMaxForward(NNCIMatrixType input);

#endif //CMATRIX_NNC_ACTIVATION_LAYER_H
