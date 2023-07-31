#ifndef CMATRIX_NNC_ACTIVATION_LAYER_H
#define CMATRIX_NNC_ACTIVATION_LAYER_H

#include "nnc_matrix.h"

NNCIMatrixType NNCActivationReLUForward(NNCIMatrixType input);
NNCIMatrixType NNCActivationReLUBackward(NNCIMatrixType input, NNCIMatrixType dvalues);

NNCIMatrixType NNCActivationSoftMaxForward(NNCIMatrixType input);
NNCIMatrixType NNCActivationSoftMaxBackward(NNCIMatrixType dvalues, NNCIMatrixType softmax_output);

#endif //CMATRIX_NNC_ACTIVATION_LAYER_H
