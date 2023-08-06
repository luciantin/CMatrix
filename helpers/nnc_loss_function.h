#ifndef CMATRIX_NNC_LOSS_FUNCTION_H
#define CMATRIX_NNC_LOSS_FUNCTION_H

#include "nnc_matrix.h"
#include <math.h>

NNCIMatrixType NNCLossCCELForward(NNCIMatrixType input, NNCIMatrixType target);
NNCIMatrixType NNCLossCCELBackward(NNCIMatrixType dvalues, NNCIMatrixType target);

NNCIMatrixType NNCActivationSoftMaxLossCCELBackward(NNCIMatrixType dvalues, NNCIMatrixType target);
#endif //CMATRIX_NNC_LOSS_FUNCTION_H
