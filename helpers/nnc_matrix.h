#ifndef CMATRIX_NNC_MATRIX_H
#define CMATRIX_NNC_MATRIX_H

#include <stdbool.h>
#include "nnc_config.h"

typedef struct NNCMatrixType
{
    nnc_matrix matrix;
    nnc_uint    x;
    nnc_uint    y;
}
NNCMatrixType;

#define NNCIMatrixType NNCMatrixType*



NNCIMatrixType NNCMatrixAlloc(nnc_uint x, nnc_uint y);
NNCIMatrixType NNCMatrixAllocBaseValue(nnc_uint x, nnc_uint y, nnc_mtype base_value);
NNCIMatrixType NNCMatrixAllocRandom(nnc_uint x, nnc_uint y);

NNCIMatrixType NNCMatrixAllocLine(nnc_uint x, nnc_uint y);
NNCIMatrixType NNCMatrixAllocSum(nnc_uint x, nnc_uint y);

void NNCMatrixDeAlloc(NNCIMatrixType matrix);
void NNCMatrixPrint(NNCIMatrixType matrix);

NNCIMatrixType NNCMatrixProduct(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b);
NNCIMatrixType NNCMatrixSum(NNCIMatrixType matrix_a, NNCIMatrixType matrix_b);
NNCIMatrixType NNCMatrixSumSingle(NNCIMatrixType matrix, bool axis);
nnc_vector NNCMatrixDotProduct(NNCIMatrixType matrix, nnc_vector vector);
NNCIMatrixType NNCMatrixAddVector(NNCIMatrixType matrix, nnc_vector vector);
NNCIMatrixType NNCMatrixTranspose(NNCIMatrixType matrix);
nnc_vector NNCMatrixArgMax(NNCIMatrixType matrix);


#endif //CMATRIX_NNC_MATRIX_H
